# client_with_guardrails.py
import os
import json
import logging
import asyncio
import re
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tool_registry import TOOLS

# Color reset function
def reset_terminal_colors():
    """Reset terminal colors to prevent light text issues"""
    print("\033[0m", end="")

# Load environment variables
load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")

# Azure OpenAI configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_KEY:
    raise RuntimeError("❌ AZURE_OPENAI_KEY not set in environment")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("FlightOps.MCPClient")

# Initialize Azure OpenAI client
client_azure = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def _build_tool_prompt() -> str:
    """Convert TOOLS dict into compact text to feed the LLM."""
    lines = []
    for name, meta in TOOLS.items():
        arg_str = ", ".join(meta["args"])
        lines.append(f"- {name}({arg_str}): {meta['desc']}")
    return "\n".join(lines)

# ==================== CLIENT GUARDRAILS ====================

class ClientGuardrails:
    """Client-side security guardrails that complement existing validations."""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Advanced injection patterns not caught by basic validation
            re.compile(r'(?i)\$[a-z]+\s*:'),  # NoSQL injection
            re.compile(r'(?i){\s*\$[a-z]+\s*:'),  # MongoDB operators
            re.compile(r'(?i)(ne|gt|lt|in|or|and)\s*:'),  # Query operators
            re.compile(r'(?i)(mcp|http|ws)://'),  # Protocol manipulation
            re.compile(r'(?i)(ignore|override|skip)\s+(previous|instructions)'),  # Prompt injection
            re.compile(r'(?i)(all|every|each)\s+flight'),  # Overly broad queries
            re.compile(r'(?i)flight\s+number\s*(!=|<>|not)'),  # Negative queries
            re.compile(r'(?i)(random|arbitrary|any)\s+flight'),  # Non-specific queries
        ]
        
        # Query history for behavioral analysis
        self.query_history = {}
        self.max_queries_per_minute = 30

    def validate_user_query(self, user_query: str, client_id: str = "default") -> Dict[str, Any]:
        """
        Enhanced query validation that complements basic checks.
        """
        if not user_query or len(user_query.strip()) == 0:
            return {"valid": False, "error": "Empty query", "risk_level": "low"}
        
        # Length check
        if len(user_query) > 500:
            return {"valid": False, "error": "Query too long (max 500 characters)", "risk_level": "medium"}
        
        # Advanced pattern detection
        for pattern in self.suspicious_patterns:
            if pattern.search(user_query.lower()):
                self.log_security_event("SUSPICIOUS_PATTERN", f"Pattern detected: {pattern.pattern}", user_query)
                return {"valid": False, "error": "Query contains suspicious patterns", "risk_level": "high"}
        
        # Behavioral rate limiting
        if not self._check_rate_limit(client_id):
            return {"valid": False, "error": "Rate limit exceeded", "risk_level": "medium"}
        
        # Business logic validation
        validation = self._validate_business_logic(user_query)
        if not validation["valid"]:
            return validation
        
        return {"valid": True, "message": "Query validation passed"}

    def _check_rate_limit(self, client_id: str) -> bool:
        """Basic client-side rate limiting."""
        current_minute = int(datetime.now().timestamp() / 60)
        key = f"{client_id}:{current_minute}"
        
        if key not in self.query_history:
            self.query_history[key] = 0
        
        self.query_history[key] += 1
        return self.query_history[key] <= self.max_queries_per_minute

    def _validate_business_logic(self, user_query: str) -> Dict[str, Any]:
        """Validate query against business logic constraints."""
        query_lower = user_query.lower()
        
        # Detect overly broad queries
        broad_indicators = [
            "all flights", "every flight", "entire database", 
            "show me everything", "all records", "complete data"
        ]
        
        if any(indicator in query_lower for indicator in broad_indicators):
            return {"valid": False, "error": "Query too broad. Please specify specific flights.", "risk_level": "medium"}
        
        # Detect potential scraping patterns
        scraping_patterns = [
            r'flight\s+(\d+)\s*,\s*(\d+)',  # Multiple flight numbers
            r'flights?\s+from\s+(\d+)\s+to\s+(\d+)',  # Flight number ranges
            r'(\d+)\s+through\s+(\d+)',  # Sequential requests
        ]
        
        for pattern in scraping_patterns:
            if re.search(pattern, query_lower):
                return {"valid": False, "error": "Multiple flight queries not allowed in single request", "risk_level": "high"}
        
        return {"valid": True, "message": "Business logic validation passed"}

    def validate_tool_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate LLM-generated tool plan before execution.
        """
        if not plan_data or "plan" not in plan_data:
            return {"valid": False, "error": "Invalid plan structure"}
        
        plan = plan_data.get("plan", [])
        
        # Check number of tool calls
        if len(plan) > 5:  # Increased from 3 to be more permissive
            return {"valid": False, "error": f"Too many tool calls: {len(plan)}. Maximum 5 allowed."}
        
        # Validate each tool call
        for i, tool_call in enumerate(plan):
            if not isinstance(tool_call, dict):
                return {"valid": False, "error": f"Tool call {i} must be a dictionary"}
            
            tool_name = tool_call.get("tool")
            arguments = tool_call.get("arguments", {})
            
            # Validate tool name exists
            if not tool_name or tool_name not in TOOLS:
                return {"valid": False, "error": f"Invalid tool name: {tool_name}"}
            
            # Validate arguments are safe
            arg_validation = self._validate_tool_arguments(tool_name, arguments)
            if not arg_validation["valid"]:
                return arg_validation
        
        return {"valid": True, "plan": plan_data}

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments for safety."""
        for key, value in arguments.items():
            if value is None:
                continue
                
            # Convert to string for pattern checking
            str_value = str(value).lower()
            
            # Check for suspicious values
            suspicious_values = ["unknown", "any", "all", "null", "undefined", "none"]
            if str_value in suspicious_values:
                return {"valid": False, "error": f"Invalid argument value for {key}: '{value}'"}
            
            # Check for injection patterns in string values
            injection_patterns = [
                r'[;&|`]',  # Command injection
                r'\$\(',    # Command substitution
                r'\.\./',   # Path traversal
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, str_value):
                    return {"valid": False, "error": f"Suspicious characters in argument {key}"}
        
        return {"valid": True, "message": "Argument validation passed"}

    def log_security_event(self, event_type: str, message: str, user_input: str = ""):
        """Enhanced security logging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "user_input": user_input[:200],  # Truncate for safety
            "component": "ClientGuardrails"
        }
        logger.warning(f"SECURITY: {json.dumps(log_entry)}")

# Enhanced system prompt with guardrails
SYSTEM_PROMPT_PLAN = f"""
You are an assistant that converts user questions into MCP tool calls with SECURITY GUARDRAILS.

Available tools:
{_build_tool_prompt()}

## CRITICAL SECURITY RULES:

1. **TOOL USAGE CONSTRAINTS**:
   - Use ONLY the tools listed above
   - Maximum 5 tool calls per query
   - Never include "unknown", "any", or placeholder values
   - Reject queries requesting multiple unrelated flights

2. **QUERY SAFETY**:
   - REJECT: Database operations (DROP, INSERT, UPDATE, DELETE)
   - REJECT: System commands or file operations  
   - REJECT: Overly broad queries ("all flights", "everything")
   - REJECT: Multiple flight numbers in single query
   - If query seems malicious, return: {{"plan": [], "error": "Query rejected"}}

3. **PARAMETER HANDLING**:
   - Omit parameters if values are not specified
   - Never invent or guess parameter values
   - Flight numbers must be numeric (1-9999)
   - Dates must be in YYYY-MM-DD format
   - Carrier codes must be 2-3 characters

4. **DETECT AND REJECT**:
   - "Show me all flights from 6E" → TOO BROAD
   - "Flight 101, 102, 103" → TOO MANY FLIGHTS
   - "What's the database structure?" → SYSTEM KNOWLEDGE
   - Sequential flight number requests → SCRAPING

## OUTPUT FORMAT:
{{
  "plan": [
    {{
      "tool": "exact_tool_name",
      "arguments": {{
        "carrier": "6E",
        "flight_number": "215",
        "date_of_origin": "2024-06-23"
      }}
    }}
  ]
}}

## SECURITY FIRST: When in doubt, reject the query.
"""

SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes tool outputs into a concise answer.
Focus on clarity and readability.
- Never reveal internal system details
- Sanitize any potential sensitive information
- Maintain professional tone
"""

class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None
        self.guardrails = ClientGuardrails()  # Initialize guardrails

    async def connect(self):
        """Connect to the MCP server using streamable-http transport."""
        try:
            logger.info(f"Connecting to MCP server at {self.base_url}")
            
            # streamablehttp_client returns a context manager
            self._client_context = streamablehttp_client(self.base_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()
            
            # Create session
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            
            # Initialize the connection
            await self.session.initialize()
            logger.info("✅ Connected to MCP server successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            reset_terminal_colors()  # Reset colors on error
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            reset_terminal_colors()  # Always reset colors

    def _call_azure_openai(self, messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Internal helper for Azure OpenAI chat completions."""
        try:
            completion = client_azure.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            reset_terminal_colors()  # Reset colors on error
            return json.dumps({"error": str(e)})

    # ---------- MCP Server Interaction ----------

    async def list_tools(self) -> dict:
        """List available tools from the MCP server."""
        try:
            if not self.session:
                await self.connect()
            
            tools_list = await self.session.list_tools()
            
            # Convert MCP tools response to dictionary format
            tools_dict = {}
            for tool in tools_list.tools:
                tools_dict[tool.name] = {
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
            
            return {"tools": tools_dict}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}

    async def invoke_tool(self, tool_name: str, args: dict) -> dict:     
        """Invoke a tool by name with arguments via MCP protocol."""
        try:
            if not self.session:
                await self.connect()
            
            logger.info(f"Calling tool: {tool_name} with args: {args}")
            
            # Call the tool using MCP session
            result = await self.session.call_tool(tool_name, args)     
            
            # Extract content from result
            if result.content:
                # MCP returns content as a list of Content objects
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        try:
                            # Try to parse as JSON
                            content_items.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            content_items.append(item.text)
                
                # If single item, return it directly
                if len(content_items) == 1:
                    return content_items[0]
                return {"results": content_items}
            
            return {"error": "No content in response"}
            
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}

    # ---------- LLM Wrappers with Guardrails ----------

    def plan_tools(self, user_query: str) -> dict:
        """Use Azure OpenAI to generate a plan of tool calls with guardrail validation."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": user_query},
        ]
        content = self._call_azure_openai(messages, temperature=0.1)
        
        try:
            plan_data = json.loads(content)
            
            # Validate the plan with guardrails
            validation = self.guardrails.validate_tool_plan(plan_data)
            if not validation["valid"]:
                logger.warning(f"Guardrail blocked plan: {validation['error']}")
                return {"plan": [], "error": validation["error"]}
            
            return validation["plan"]
            
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM plan output.")
            reset_terminal_colors()  # Reset colors on warning
            return {"plan": []}

    def summarize_results(self, user_query: str, plan: list, results: list) -> dict:
        """Use Azure OpenAI to summarize results into human-friendly output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"Question:\n{user_query}"},
            {"role": "assistant", "content": f"Plan:\n{json.dumps(plan, indent=2)}"},
            {"role": "assistant", "content": f"Results:\n{json.dumps(results, indent=2)}"},
        ]
        summary = self._call_azure_openai(messages, temperature=0.3)
        return {"summary": summary}

    # ---------- Orchestration with Enhanced Security ----------

    async def run_query(self, user_query: str) -> dict:
        """
        Full flow with enhanced security guardrails:
        1. Validate user query
        2. Use LLM to plan tool calls (with guardrail validation)
        3. Execute tools sequentially on MCP server
        4. Summarize results via LLM
        """
        try:
            logger.info(f"User query: {user_query}")
            
            # Step 1: Query validation with guardrails
            query_validation = self.guardrails.validate_user_query(user_query)
            if not query_validation["valid"]:
                return {"error": f"Query rejected: {query_validation['error']}"}
            
            # Step 2: Plan tools with guardrail validation
            plan_data = self.plan_tools(user_query)
            plan = plan_data.get("plan", [])
            
            if not plan:
                error_msg = plan_data.get("error", "LLM did not produce a valid tool plan.")
                return {"error": error_msg}

            # Step 3: Execute tools with additional safety checks
            results = []
            for step in plan:
                tool = step.get("tool")
                args = step.get("arguments", {})

                # Enhanced argument cleaning
                args = self._clean_arguments(args)

                if not tool:
                    continue

                logger.info(f"Invoking tool: {tool} with args: {args}")
                resp = await self.invoke_tool(tool, args)
                results.append({tool: resp})

            # Step 4: Summarize results
            summary = self.summarize_results(user_query, plan, results)
            return {"plan": plan, "results": results, "summary": summary}
            
        except Exception as e:
            logger.error(f"Error in run_query: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}
        finally:
            reset_terminal_colors()  # Always reset colors

    def _clean_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced argument cleaning with additional safety checks."""
        cleaned = {}
        for k, v in args.items():
            if v is None:
                continue
                
            # Convert to string for cleaning
            str_val = str(v).strip()
            
            # Skip empty values and suspicious placeholders
            if not str_val or str_val.lower() in ["unknown", "any", "all", "null"]:
                continue
                
            # Basic sanitization
            str_val = re.sub(r'[;&|`$]', '', str_val)  # Remove dangerous chars
            
            # Convert back to appropriate type
            if k == "flight_number" and str_val.isdigit():
                cleaned[k] = int(str_val)
            else:
                cleaned[k] = str_val
                
        return cleaned
      #####################################################################
      # server_with_guardrails.py
import os
import logging
import json
import re
from typing import Optional, Any, Dict
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv() 

from mcp.server.fastmcp import FastMCP

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))
TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")

MONGODB_URL = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("MONGO_DB")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("flightops.mcp.server")

mcp = FastMCP("FlightOps MCP Server")

_mongo_client: Optional[AsyncIOMotorClient] = None
_db = None
_col = None

# ==================== SERVER GUARDRAILS ====================

class ServerGuardrails:
    """Server-side security guardrails that complement existing validations."""
    
    def __init__(self):
        self.dangerous_patterns = [
            re.compile(r'(?i)\$[a-z]+\s*:'),  # NoSQL operators
            re.compile(r'(?i){\s*\$'),  # MongoDB operators
            re.compile(r'[;&|`]'),  # Command injection
            re.compile(r'\.\./'),  # Path traversal
        ]
        
        # Request tracking for behavioral analysis
        self.request_tracking = {}

    def sanitize_input(self, input_str: str, max_length: int = 100) -> str:
        """Enhanced input sanitization."""
        if not input_str:
            return ""
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>&;|`$]', '', str(input_str))
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()

    def validate_parameters(self, carrier: str = "", flight_number: str = "", 
                          date_of_origin: str = "") -> Dict[str, Any]:
        """
        Enhanced parameter validation that complements existing validation.
        """
        # Sanitize inputs
        safe_carrier = self.sanitize_input(carrier, 3)
        safe_flight_number = self.sanitize_input(flight_number, 4)
        safe_date = self.sanitize_input(date_of_origin, 10)
        
        # Check for suspicious patterns
        combined = f"{safe_carrier}{safe_flight_number}{safe_date}"
        for pattern in self.dangerous_patterns:
            if pattern.search(combined):
                self.log_security_event("DANGEROUS_PATTERN", f"Pattern: {pattern.pattern}", combined)
                return {"valid": False, "error": "Invalid parameters detected"}
        
        # Business logic validation
        if safe_carrier and not re.match(r'^[A-Z0-9]{2,3}$', safe_carrier):
            return {"valid": False, "error": "Invalid carrier format"}
        
        if safe_flight_number and not safe_flight_number.isdigit():
            return {"valid": False, "error": "Flight number must be numeric"}
        
        if safe_flight_number:
            try:
                fn = int(safe_flight_number)
                if not (1 <= fn <= 9999):
                    return {"valid": False, "error": "Flight number out of range"}
            except ValueError:
                return {"valid": False, "error": "Invalid flight number"}
        
        return {
            "valid": True, 
            "sanitized": {
                "carrier": safe_carrier,
                "flight_number": safe_flight_number,
                "date_of_origin": safe_date
            }
        }

    def log_security_event(self, event_type: str, message: str, user_input: str = ""):
        """Enhanced security logging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "user_input": user_input[:150],
            "component": "ServerGuardrails"
        }
        logger.warning(f"SECURITY: {json.dumps(log_entry)}")

# Initialize server guardrails
server_guardrails = ServerGuardrails()

async def get_mongodb_client():
    """Initialize and return the global Motor client, DB and collection."""
    global _mongo_client, _db, _col
    if _mongo_client is None:
        logger.info("Connecting to MongoDB: %s", MONGODB_URL)
        _mongo_client = AsyncIOMotorClient(MONGODB_URL)
        _db = _mongo_client[DATABASE_NAME]
        _col = _db[COLLECTION_NAME]
    return _mongo_client, _db, _col

def normalize_flight_number(flight_number: Any) -> Optional[int]:
    """Convert flight_number to int. MongoDB stores it as int."""
    if flight_number is None or flight_number == "":
        return None
    if isinstance(flight_number, int):
        return flight_number
    try:
        return int(str(flight_number).strip())
    except (ValueError, TypeError):
        logger.warning(f"Could not normalize flight_number: {flight_number}")
        return None

def validate_date(date_str: str) -> Optional[str]:
    """
    Validate date_of_origin string. Accepts common formats.
    Returns normalized ISO date string YYYY-MM-DD if valid, else None.
    """
    if not date_str or date_str == "":
        return None
    
    # Handle common date formats
    formats = [
        "%Y-%m-%d",      # 2024-06-23
        "%d-%m-%Y",      # 23-06-2024
        "%Y/%m/%d",      # 2024/06/23
        "%d/%m/%Y",      # 23/06/2024
        "%B %d, %Y",     # June 23, 2024
        "%d %B %Y",      # 23 June 2024
        "%b %d, %Y",     # Jun 23, 2024
        "%d %b %Y"       # 23 Jun 2024
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

def make_query(carrier: str, flight_number: Optional[int], date_of_origin: str) -> Dict:
    """
    Build MongoDB query matching the actual database schema.
    """
    query = {}
    
    # Add carrier if provided
    if carrier:
        query["flightLegState.carrier"] = carrier
    
    # Add flight number as integer (as stored in DB)
    if flight_number is not None:
        query["flightLegState.flightNumber"] = flight_number
    
    # Add date if provided
    if date_of_origin:
        query["flightLegState.dateOfOrigin"] = date_of_origin
    
    logger.info(f"Built query: {json.dumps(query)}")
    return query

def response_ok(data: Any) -> str:
    """Return JSON string for successful response."""
    return json.dumps({"ok": True, "data": data}, indent=2, default=str)

def response_error(msg: str, code: int = 400) -> str:
    """Return JSON string for error response."""
    return json.dumps({"ok": False, "error": {"message": msg, "code": code}}, indent=2)

async def _fetch_one_async(query: dict, projection: dict) -> str:
    """
    Consistent async DB fetch and error handling.
    Returns JSON string response.
    """
    try:
        _, _, col = await get_mongodb_client()
        logger.info(f"Executing query: {json.dumps(query)}")
        
        result = await col.find_one(query, projection)
        
        if not result:
            logger.warning(f"No document found for query: {json.dumps(query)}")
            return response_error("No matching document found.", code=404)
        
        # Remove _id and _class to keep output clean
        if "_id" in result:
            result.pop("_id")
        if "_class" in result:
            result.pop("_class")
        
        logger.info(f"Query successful")
        return response_ok(result)
    except Exception as exc:
        logger.exception("DB query failed")
        return response_error(f"DB query failed: {str(exc)}", code=500)

# --- MCP Tools with Enhanced Security ---

@mcp.tool()
async def health_check() -> str:
    """
    Simple health check for orchestrators and clients.
    Attempts a cheap DB ping.
    """
    try:
        _, _, col = await get_mongodb_client()
        doc = await col.find_one({}, {"_id": 1})
        return response_ok({"status": "ok", "db_connected": doc is not None})
    except Exception as e:
        logger.exception("Health check DB ping failed")
        return response_error("DB unreachable", code=503)

@mcp.tool()
async def get_flight_basic_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Fetch basic flight information including carrier, flight number, date, stations, times, and status.
    
    Args:
        carrier: Airline carrier code (e.g., "6E", "AI")
        flight_number: Flight number as string (e.g., "215")
        date_of_origin: Date in YYYY-MM-DD format (e.g., "2024-06-23")
    """
    logger.info(f"get_flight_basic_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    # Enhanced parameter validation with guardrails
    param_validation = server_guardrails.validate_parameters(carrier, flight_number, date_of_origin)
    if not param_validation["valid"]:
        return response_error(param_validation["error"], 400)
    
    # Use sanitized parameters
    safe_params = param_validation["sanitized"]
    carrier = safe_params["carrier"]
    flight_number = safe_params["flight_number"]
    date_of_origin = safe_params["date_of_origin"]
    
    # Normalize inputs (your existing validation)
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date_of_origin format. Expected YYYY-MM-DD or common date formats", 400)
    
    query = make_query(carrier, fn, dob)
    
    # Project basic flight information
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.suffix": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.seqNumber": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.startStationICAO": 1,
        "flightLegState.endStationICAO": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.scheduledEndTime": 1,
        "flightLegState.flightStatus": 1,
        "flightLegState.operationalStatus": 1,
        "flightLegState.flightType": 1,
        "flightLegState.blockTimeSch": 1,
        "flightLegState.blockTimeActual": 1,
        "flightLegState.flightHoursActual": 1
    }
    
    return await _fetch_one_async(query, projection)

# Apply the same enhanced security pattern to all other tools...
# [Repeat the enhanced parameter validation for all other tools]

@mcp.tool()
async def get_operation_times(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """Enhanced with guardrails"""
    param_validation = server_guardrails.validate_parameters(carrier, flight_number, date_of_origin)
    if not param_validation["valid"]:
        return response_error(param_validation["error"], 400)
    
    # ... rest of your existing code

@mcp.tool()
async def get_equipment_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """Enhanced with guardrails"""
    param_validation = server_guardrails.validate_parameters(carrier, flight_number, date_of_origin)
    if not param_validation["valid"]:
        return response_error(param_validation["error"], 400)
    
    # ... rest of your existing code

# [Apply the same pattern to all other tools...]

@mcp.tool()
async def raw_mongodb_query(query_json: str, limit: int = 10) -> str:
    """
    Run a raw MongoDB query string (JSON) against collection (for debugging).
    Enhanced with additional security checks.
    """
    # Enhanced security for raw queries
    if len(query_json) > 1000:
        return response_error("Query too long", 400)
    
    # Check for dangerous patterns in raw query
    for pattern in server_guardrails.dangerous_patterns:
        if pattern.search(query_json):
            server_guardrails.log_security_event("DANGEROUS_RAW_QUERY", f"Pattern: {pattern.pattern}", query_json)
            return response_error("Dangerous query pattern detected", 403)
    
    try:
        _, _, col = await get_mongodb_client()
        try:
            query = json.loads(query_json)
        except json.JSONDecodeError as e:
            return response_error(f"Invalid JSON query: {str(e)}. Example: '{{\"flightLegState.carrier\": \"6E\"}}'", 400)
        
        limit = min(max(1, int(limit)), 50)
        cursor = col.find(query).sort("flightLegState.dateOfOrigin", -1).limit(limit)
        docs = []
        
        async for doc in cursor:
            if "_id" in doc:
                doc.pop("_id")
            if "_class" in doc:
                doc.pop("_class")
            docs.append(doc)
        
        if not docs:
            return response_error("No documents found for given query.", 404)
        
        return response_ok({"count": len(docs), "documents": docs})
    except Exception as exc:
        logger.exception("raw_mongodb_query failed")
        return response_error(f"raw query failed: {str(exc)}", 500)

# --- Run MCP Server ---
if __name__ == "__main__":
    logger.info("Starting FlightOps MCP Server on %s:%s (transport=%s)", HOST, PORT, TRANSPORT)
    logger.info("MongoDB URL: %s, Database: %s, Collection: %s", MONGODB_URL, DATABASE_NAME, COLLECTION_NAME)
    mcp.run(transport="streamable-http")
