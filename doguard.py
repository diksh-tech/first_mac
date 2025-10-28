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
    """Client-side guardrails that only block clear CRUD attempts."""
    
    def __init__(self):
        # Only block clear modification attempts
        self.blocked_patterns = [
            # Clear modification attempts
            re.compile(r'(?i)^\s*(create|insert|add|update|modify|delete|remove|drop)\s+'),
            re.compile(r'(?i)\b(create|insert|add|update|modify|delete|remove|drop)\s+(new\s+)?flight\b'),
            re.compile(r'(?i)\b(book|reserve|cancel|change|edit)\s+flight\b'),
            re.compile(r'(?i)\b(set|assign)\s+flight\s+status\b'),
            
            # Database operations
            re.compile(r'(?i)\b(drop\s+table|truncate|alter\s+table)\b'),
            re.compile(r'(?i)\b(backup|restore|import|export)\s+database\b'),
        ]
        
        # Dangerous technical patterns
        self.dangerous_patterns = [
            re.compile(r'(?i)\$[a-z]+\s*:'),  # NoSQL injection
            re.compile(r'[;&|`]'),  # Command injection
        ]

    def validate_user_query(self, user_query: str) -> Dict[str, Any]:
        """
        Validate user query - only block clear CRUD attempts.
        """
        if not user_query or len(user_query.strip()) == 0:
            return {"valid": False, "error": "Empty query"}
        
        # Check length
        if len(user_query) > 1000:
            return {"valid": False, "error": "Query too long (max 1000 characters)"}
        
        query_lower = user_query.lower().strip()
        
        # Check for dangerous technical patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(query_lower):
                logger.warning(f"Blocked dangerous pattern: {pattern.pattern}")
                return {"valid": False, "error": "Query contains dangerous patterns"}
        
        # Check for clear CRUD attempts (only block obvious ones)
        for pattern in self.blocked_patterns:
            if pattern.search(query_lower):
                logger.warning(f"Blocked CRUD attempt: {pattern.pattern}")
                return {"valid": False, "error": "Modification operations are not allowed. This is a read-only system."}
        
        # ALLOW normal flight queries
        return {"valid": True, "message": "Query validation passed"}

# Enhanced system prompt for consistent JSON output
SYSTEM_PROMPT_PLAN = f"""
You are a flight information assistant that converts user questions into MCP tool calls.

Available tools:
{_build_tool_prompt()}

## CRITICAL INSTRUCTIONS:
1. **OUTPUT ONLY RAW JSON** - no other text, no explanations, no markdown
2. Always use this exact structure:
{{
  "plan": [
    {{
      "tool": "tool_name",
      "arguments": {{
        "carrier": "6E",
        "flight_number": "215",
        "date_of_origin": "2024-06-23"
      }}
    }}
  ]
}}

3. **Tool Selection Rules**:
   - For general flight info: use get_flight_basic_info
   - For timing/schedules: use get_operation_times  
   - For aircraft details: use get_equipment_info
   - For delays: use get_delay_summary
   - For fuel: use get_fuel_summary
   - For passengers: use get_passenger_info
   - For crew: use get_crew_info

4. **Parameter Rules**:
   - Omit parameters if not specified (don't use "unknown")
   - Flight numbers should be strings (e.g., "215")
   - Dates should be in YYYY-MM-DD format
   - Carrier codes are 2-3 letters (e.g., "6E")

## EXAMPLES:

User: "flight 6E 215 on 2024-06-23"
Output:
{{
  "plan": [
    {{
      "tool": "get_flight_basic_info",
      "arguments": {{
        "carrier": "6E",
        "flight_number": "215",
        "date_of_origin": "2024-06-23"
      }}
    }}
  ]
}}

User: "show delays for flight 6E 215"
Output:
{{
  "plan": [
    {{
      "tool": "get_delay_summary", 
      "arguments": {{
        "carrier": "6E",
        "flight_number": "215"
      }}
    }}
  ]
}}

User: "what aircraft for 6E 215"
Output:
{{
  "plan": [
    {{
      "tool": "get_equipment_info",
      "arguments": {{
        "carrier": "6E", 
        "flight_number": "215"
      }}
    }}
  ]
}}

**REMEMBER: Output ONLY the JSON, nothing else!**
"""

SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes flight data into clear, concise answers.
Focus on clarity and readability.
Never suggest modifications or actions - just report the facts.
"""

class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None
        self.guardrails = ClientGuardrails()

    async def connect(self):
        """Connect to the MCP server using streamable-http transport."""
        try:
            logger.info(f"Connecting to MCP server at {self.base_url}")
            
            self._client_context = streamablehttp_client(self.base_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()
            
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            
            await self.session.initialize()
            logger.info("✅ Connected to MCP server successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            reset_terminal_colors()
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
            reset_terminal_colors()

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
            reset_terminal_colors()
            return json.dumps({"error": str(e)})

    # ---------- MCP Server Interaction ----------

    async def list_tools(self) -> dict:
        """List available tools from the MCP server."""
        try:
            if not self.session:
                await self.connect()
            
            tools_list = await self.session.list_tools()
            
            tools_dict = {}
            for tool in tools_list.tools:
                tools_dict[tool.name] = {
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
            
            return {"tools": tools_dict}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            reset_terminal_colors()
            return {"error": str(e)}

    async def invoke_tool(self, tool_name: str, args: dict) -> dict:     
        """Invoke a tool by name with arguments via MCP protocol."""
        try:
            if not self.session:
                await self.connect()
            
            logger.info(f"Calling tool: {tool_name} with args: {args}")
            
            result = await self.session.call_tool(tool_name, args)     
            
            if result.content:
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        try:
                            content_items.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            content_items.append(item.text)
                
                if len(content_items) == 1:
                    return content_items[0]
                return {"results": content_items}
            
            return {"error": "No content in response"}
            
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            reset_terminal_colors()
            return {"error": str(e)}

    # ---------- LLM Wrappers ----------

    def plan_tools(self, user_query: str) -> dict:
        """Use Azure OpenAI to generate a plan of tool calls."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": user_query},
        ]
        
        content = self._call_azure_openai(messages, temperature=0.1)
        logger.info(f"Raw LLM response: {content}")
        
        # Clean and parse JSON response
        cleaned_content = self._clean_json_response(content)
        
        try:
            plan_data = json.loads(cleaned_content)
            
            # Validate the structure
            if isinstance(plan_data, dict) and "plan" in plan_data:
                if isinstance(plan_data["plan"], list):
                    return plan_data
                else:
                    logger.warning("Plan is not a list")
                    return {"plan": []}
            else:
                logger.warning("Missing 'plan' key in response")
                return {"plan": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Failed to parse: {cleaned_content}")
            return {"plan": []}

    def _clean_json_response(self, content: str) -> str:
        """Clean and extract JSON from LLM response."""
        if not content:
            return "{}"
        
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Find first { and last }
        start = content.find('{')
        end = content.rfind('}') + 1
        
        if start != -1 and end != 0 and end > start:
            json_str = content[start:end]
            return json_str
        else:
            return content.strip()

    def summarize_results(self, user_query: str, plan: list, results: list) -> dict:
        """Use Azure OpenAI to summarize results into human-friendly output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"Question: {user_query}"},
            {"role": "assistant", "content": f"Tool Plan: {json.dumps(plan, indent=2)}"},
            {"role": "assistant", "content": f"Tool Results: {json.dumps(results, indent=2)}"},
        ]
        summary = self._call_azure_openai(messages, temperature=0.3)
        return {"summary": summary}

    # ---------- Orchestration ----------

    async def run_query(self, user_query: str) -> dict:
        """
        Full flow with guardrails:
        1. Validate user query (only block clear threats)
        2. Use LLM to plan tool calls  
        3. Execute tools on MCP server
        4. Summarize results
        """
        try:
            logger.info(f"User query: {user_query}")
            
            # Step 1: Query validation
            query_validation = self.guardrails.validate_user_query(user_query)
            if not query_validation["valid"]:
                return {"error": f"Query rejected: {query_validation['error']}"}
            
            # Step 2: Plan tools
            plan_data = self.plan_tools(user_query)
            plan = plan_data.get("plan", [])

            if not plan:
                return {"error": "Could not generate a valid tool plan for your query."}

            # Step 3: Execute tools
            results = []
            for step in plan:
                tool = step.get("tool")
                args = step.get("arguments", {})

                # Clean arguments
                args = {
                    k: v for k, v in args.items()
                    if v is not None and str(v).strip() != "" and str(v).lower() != "unknown"
                }

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
            reset_terminal_colors()
            return {"error": str(e)}
        finally:
            reset_terminal_colors()

# Main execution
if __name__ == "__main__":
    # Example usage
    client = FlightOpsMCPClient()
    
    async def test_query():
        result = await client.run_query("flight 6E 215 on 2024-06-23")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_query())
  ######################################################################################
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
    """Server-side protection for raw queries only."""
    
    def __init__(self):
        # Block clear database modification patterns
        self.blocked_patterns = [
            re.compile(r'(?i)\$[a-z]+\s*:'),  # NoSQL operators
            re.compile(r'[;&|`]'),  # Command injection
        ]
        
        # CRUD operation patterns in raw queries
        self.crud_patterns = [
            re.compile(r'(?i)\$(set|push|pull|unset|inc|rename)\s*:'),
            re.compile(r'(?i)(updateOne|updateMany|insertOne|insertMany|deleteOne|deleteMany)\s*\('),
        ]

    def validate_raw_query(self, query_json: str) -> Dict[str, Any]:
        """
        Validate raw queries for security threats.
        """
        # Check for dangerous patterns
        for pattern in self.blocked_patterns:
            if pattern.search(query_json):
                logger.warning(f"Blocked dangerous pattern in raw query: {pattern.pattern}")
                return {"valid": False, "error": "Dangerous query pattern detected"}
        
        # Check for CRUD operations
        for pattern in self.crud_patterns:
            if pattern.search(query_json):
                logger.warning(f"Blocked CRUD operation in raw query: {pattern.pattern}")
                return {"valid": False, "error": "Modification operations are not allowed"}
        
        return {"valid": True, "message": "Query validation passed"}

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
    
    if carrier:
        query["flightLegState.carrier"] = carrier
    
    if flight_number is not None:
        query["flightLegState.flightNumber"] = flight_number
    
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
        
        if "_id" in result:
            result.pop("_id")
        if "_class" in result:
            result.pop("_class")
        
        logger.info(f"Query successful")
        return response_ok(result)
    except Exception as exc:
        logger.exception("DB query failed")
        return response_error(f"DB query failed: {str(exc)}", code=500)

# --- MCP Tools ---

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
    
    # Normalize inputs
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

@mcp.tool()
async def get_operation_times(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Return estimated and actual operation times for a flight including takeoff, landing, block times.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_operation_times: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date format.", 400)
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.scheduledEndTime": 1,
        "flightLegState.operation.estimatedTimes": 1,
        "flightLegState.operation.actualTimes": 1,
        "flightLegState.taxiOutTime": 1,
        "flightLegState.taxiInTime": 1,
        "flightLegState.blockTimeSch": 1,
        "flightLegState.blockTimeActual": 1,
        "flightLegState.flightHoursActual": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_equipment_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Get aircraft equipment details including aircraft type, registration (tail number), and configuration.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_equipment_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.equipment.plannedAircraftType": 1,
        "flightLegState.equipment.aircraft": 1,
        "flightLegState.equipment.aircraftConfiguration": 1,
        "flightLegState.equipment.aircraftRegistration": 1,
        "flightLegState.equipment.assignedAircraftTypeIATA": 1,
        "flightLegState.equipment.assignedAircraftTypeICAO": 1,
        "flightLegState.equipment.assignedAircraftTypeIndigo": 1,
        "flightLegState.equipment.assignedAircraftConfiguration": 1,
        "flightLegState.equipment.tailLock": 1,
        "flightLegState.equipment.onwardFlight": 1,
        "flightLegState.equipment.actualOnwardFlight": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_delay_summary(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Summarize delay reasons, durations, and total delay time for a specific flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_delay_summary: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.operation.actualTimes.offBlock": 1,
        "flightLegState.delays": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_fuel_summary(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Retrieve fuel summary including planned vs actual fuel for takeoff, landing, and total consumption.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_fuel_summary: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.operation.fuel": 1,
        "flightLegState.operation.flightPlan.offBlockFuel": 1,
        "flightLegState.operation.flightPlan.takeoffFuel": 1,
        "flightLegState.operation.flightPlan.landingFuel": 1,
        "flightLegState.operation.flightPlan.holdFuel": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_passenger_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Get passenger count and connection information for the flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_passenger_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOf_origin": 1,
        "flightLegState.pax": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_crew_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Get crew connections and details for the flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_crew_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.crewConnections": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def raw_mongodb_query(query_json: str, limit: int = 10) -> str:
    """
    Run a raw MongoDB query string (JSON) against collection (for debugging).
    Returns up to `limit` documents.
    
    Args:
        query_json: MongoDB query as JSON string (e.g., '{"flightLegState.carrier": "6E"}')
        limit: Maximum number of documents to return (default 10, max 50)
    """
    # Security validation for raw queries
    query_validation = server_guardrails.validate_raw_query(query_json)
    if not query_validation["valid"]:
        return response_error(query_validation["error"], 403)
    
    # Length check
    if len(query_json) > 2000:
        return response_error("Query too long", 400)
    
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
