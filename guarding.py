# client.py (with CRUD blocking)
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

# ==================== ENHANCED CLIENT GUARDRAILS WITH CRUD BLOCKING ====================

class ClientGuardrails:
    """Enhanced client-side guardrails with CRUD operation blocking."""
    
    def __init__(self):
        # Critical patterns that could cause harm
        self.critical_patterns = [
            re.compile(r'(?i)\$[a-z]+\s*:'),  # NoSQL injection
            re.compile(r'[;&|`]'),  # Command injection
            re.compile(r'(?i)(drop\s+table|delete\s+from|insert\s+into)'),  # SQL injection
        ]
        
        # CRUD operation patterns - BLOCK THESE
        self.crud_patterns = [
            # Create operations
            re.compile(r'(?i)\b(create|insert|add|new)\b'),
            re.compile(r'(?i)\b(save|store|write)\b.*\b(flight|data|record)\b'),
            re.compile(r'(?i)\b(register|book|reserve)\b.*\b(flight|ticket|seat)\b'),
            
            # Update operations
            re.compile(r'(?i)\b(update|modify|change|edit|alter)\b'),
            re.compile(r'(?i)\b(set|assign)\b.*\b(flight|status|time)\b'),
            re.compile(r'(?i)\b(cancel|delay|reschedule)\b.*\b(flight)\b'),
            
            # Delete operations
            re.compile(r'(?i)\b(delete|remove|erase|drop|clear)\b'),
            re.compile(r'(?i)\b(remove|delete)\b.*\b(flight|record|data)\b'),
            
            # Database operations
            re.compile(r'(?i)\b(truncate|purge|clean)\b.*\b(table|database)\b'),
            re.compile(r'(?i)\b(backup|restore|import|export)\b'),
        ]
        
        # Read operations patterns (ALLOWED - your current functionality)
        self.allowed_read_patterns = [
            re.compile(r'(?i)\b(get|fetch|find|search|query|retrieve|show|display|list)\b'),
            re.compile(r'(?i)\b(details|information|info|status|summary)\b'),
            re.compile(r'(?i)\b(flight|carrier|airline|aircraft|equipment)\b'),
            re.compile(r'(?i)\b(delay|time|schedule|operation|fuel|passenger|crew)\b'),
        ]
        
        # Basic rate limiting
        self.query_count = 0
        self.last_reset = datetime.now()

    def validate_user_query(self, user_query: str) -> Dict[str, Any]:
        """
        Enhanced validation - block CRUD operations and dangerous queries.
        """
        if not user_query or len(user_query.strip()) == 0:
            return {"valid": False, "error": "Empty query"}
        
        # Check length
        if len(user_query) > 1000:
            return {"valid": False, "error": "Query too long (max 1000 characters)"}
        
        # Check for critical security patterns
        for pattern in self.critical_patterns:
            if pattern.search(user_query.lower()):
                self._log_security_event("CRITICAL_PATTERN", f"Blocked: {pattern.pattern}", user_query)
                return {"valid": False, "error": "Query contains dangerous patterns"}
        
        # Check for CRUD operations - BLOCK THESE
        crud_detected = self._detect_crud_operations(user_query)
        if crud_detected["detected"]:
            self._log_security_event("CRUD_OPERATION", f"Blocked CRUD: {crud_detected['type']}", user_query)
            return {"valid": False, "error": f"Modification operations are not allowed. {crud_detected['message']}"}
        
        # Validate this is a read-only query
        if not self._is_read_only_query(user_query):
            return {"valid": False, "error": "Only read operations are allowed. Use queries like 'get flight details', 'show flight status', etc."}
        
        # Basic rate limiting (reset every minute)
        current_time = datetime.now()
        if (current_time - self.last_reset).seconds > 60:
            self.query_count = 0
            self.last_reset = current_time
        
        self.query_count += 1
        if self.query_count > 100:  # 100 queries per minute
            return {"valid": False, "error": "Rate limit exceeded"}
        
        return {"valid": True, "message": "Query validation passed"}

    def _detect_crud_operations(self, user_query: str) -> Dict[str, Any]:
        """Detect and classify CRUD operations in user queries."""
        query_lower = user_query.lower()
        
        for pattern in self.crud_patterns:
            if pattern.search(query_lower):
                pattern_text = pattern.pattern
                
                # Classify the type of CRUD operation
                if any(op in pattern_text for op in ['create', 'insert', 'add', 'new', 'save', 'store']):
                    return {
                        "detected": True,
                        "type": "CREATE",
                        "message": "Creating new records is not permitted."
                    }
                elif any(op in pattern_text for op in ['update', 'modify', 'change', 'edit', 'alter', 'set', 'assign']):
                    return {
                        "detected": True,
                        "type": "UPDATE", 
                        "message": "Modifying existing records is not permitted."
                    }
                elif any(op in pattern_text for op in ['delete', 'remove', 'erase', 'drop', 'clear']):
                    return {
                        "detected": True,
                        "type": "DELETE",
                        "message": "Deleting records is not permitted."
                    }
                else:
                    return {
                        "detected": True,
                        "type": "OTHER",
                        "message": "This operation is not permitted in read-only mode."
                    }
        
        return {"detected": False, "type": "READ", "message": ""}

    def _is_read_only_query(self, user_query: str) -> bool:
        """Validate that the query is read-only."""
        query_lower = user_query.lower()
        
        # Check if query contains at least one allowed read pattern
        has_read_pattern = any(pattern.search(query_lower) for pattern in self.allowed_read_patterns)
        
        # Common flight query patterns that should be allowed
        flight_query_indicators = [
            'flight', 'carrier', 'airline', 'aircraft', 'schedule',
            'status', 'time', 'delay', 'fuel', 'passenger', 'crew',
            'equipment', 'operation'
        ]
        
        has_flight_context = any(indicator in query_lower for indicator in flight_query_indicators)
        
        return has_read_pattern and has_flight_context

    def validate_tool_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool plan to ensure no CRUD operations are attempted.
        """
        if not plan_data or "plan" not in plan_data:
            return {"valid": False, "error": "Invalid plan structure"}
        
        plan = plan_data.get("plan", [])
        
        # Allow reasonable number of tool calls
        if len(plan) > 10:
            return {"valid": False, "error": "Too many tool calls (max 10)"}
        
        # Validate each tool call exists in registry and is read-only
        for i, tool_call in enumerate(plan):
            if not isinstance(tool_call, dict):
                return {"valid": False, "error": f"Tool call {i} must be a dictionary"}
            
            tool_name = tool_call.get("tool")
            if not tool_name or tool_name not in TOOLS:
                return {"valid": False, "error": f"Invalid tool name: {tool_name}"}
            
            # Additional validation for tool arguments
            args_validation = self._validate_tool_arguments(tool_name, tool_call.get("arguments", {}))
            if not args_validation["valid"]:
                return args_validation
        
        return {"valid": True, "plan": plan_data}

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments for safety and read-only compliance."""
        for key, value in arguments.items():
            if value is None:
                continue
                
            # Convert to string for pattern checking
            str_value = str(value).lower()
            
            # Check for suspicious values that might indicate modification intent
            suspicious_values = ["unknown", "any", "all", "null", "undefined", "none", "new", "latest"]
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

    def _log_security_event(self, event_type: str, message: str, user_input: str = ""):
        """Enhanced security logging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "user_input": user_input[:200],
            "component": "ClientGuardrails"
        }
        logger.warning(f"SECURITY_BLOCKED: {json.dumps(log_entry)}")

# Enhanced system prompt with explicit CRUD blocking
SYSTEM_PROMPT_PLAN = f"""
You are an assistant that converts user questions into MCP tool calls.

Available tools (READ-ONLY):
{_build_tool_prompt()}

## SECURITY RULES (STRICTLY ENFORCED):

1. **READ-ONLY OPERATIONS ONLY**:
   - This is a READ-ONLY system - no modifications allowed
   - BLOCK any requests to: create, update, delete, or modify data
   - BLOCK: flight bookings, cancellations, status changes, data updates
   - ONLY allow: querying, searching, retrieving flight information

2. **ALLOWED OPERATIONS**:
   - Get flight details, status, schedules
   - Query flight delays, equipment, fuel, passengers, crew
   - Search flight information by carrier, flight number, date
   - View operational times and summaries

3. **BLOCKED OPERATIONS**:
   - ❌ "Book a flight", "Create new flight", "Add passenger"
   - ❌ "Update flight status", "Change schedule", "Modify equipment"
   - ❌ "Cancel flight", "Delete record", "Remove passenger"
   - ❌ Any other CREATE, UPDATE, DELETE operations

4. **TOOL USAGE**:
   - Use ONLY the tools listed above
   - Never include "unknown", "any", or placeholder values
   - Omit parameters if values are not specified

## OUTPUT FORMAT:
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

## RESPONSE FOR BLOCKED QUERIES:
If a query attempts any modification, return:
{{
  "plan": [],
  "error": "Read-only system: Modification operations are not permitted"
}}
"""

SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes tool outputs into a concise answer.
Focus on clarity and readability.

IMPORTANT: Never suggest or imply that modifications can be made.
Use phrases like:
- "The flight information shows..."
- "According to the available data..."
- "The records indicate..."
- "Based on the query results..."

Never use phrases like:
- "You can update..." 
- "To change this..."
- "You should modify..."
- "The system allows you to..."
"""

class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None
        self.guardrails = ClientGuardrails()  # Initialize guardrails

    # [ALL YOUR EXISTING METHODS REMAIN EXACTLY THE SAME]
    # connect(), disconnect(), _call_azure_openai(), list_tools(), invoke_tool()
    # plan_tools(), summarize_results() - all remain unchanged

    async def run_query(self, user_query: str) -> dict:
        """
        Full flow with CRUD operation blocking:
        1. Validate user query (block CRUD operations)
        2. Use LLM to plan tool calls
        3. Execute tools sequentially on MCP server
        4. Summarize results via LLM
        """
        try:
            logger.info(f"User query: {user_query}")
            
            # Step 1: Enhanced query validation - BLOCK CRUD OPERATIONS
            query_validation = self.guardrails.validate_user_query(user_query)
            if not query_validation["valid"]:
                return {"error": f"Security rejection: {query_validation['error']}"}
            
            # Step 2: Plan tools (existing functionality)
            plan_data = self.plan_tools(user_query)
            plan = plan_data.get("plan", [])

            # If plan is empty due to CRUD blocking in LLM, return meaningful error
            if not plan and "error" in plan_data:
                return {"error": plan_data["error"]}
            elif not plan:
                return {"error": "No valid tool plan generated. Query may attempt unsupported operations."}

            # Step 3: Execute tools (existing functionality)
            results = []
            for step in plan:
                tool = step.get("tool")
                args = step.get("arguments", {})

                # Clean up 'unknown' or empty args (your existing logic)
                args = {
                    k: v for k, v in args.items()
                    if v is not None and str(v).strip() != "" and str(v).lower() != "unknown"
                }

                if not tool:
                    continue

                logger.info(f"Invoking tool: {tool} with args: {args}")
                resp = await self.invoke_tool(tool, args)
                results.append({tool: resp})

            # Step 4: Summarize results (existing functionality)
            summary = self.summarize_results(user_query, plan, results)
            return {"plan": plan, "results": results, "summary": summary}
            
        except Exception as e:
            logger.error(f"Error in run_query: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}
        finally:
            reset_terminal_colors()  # Always reset colors
##############################################################################################
# server.py (with CRUD protection)
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

# ==================== ENHANCED SERVER GUARDRAILS WITH CRUD PROTECTION ====================

class ServerGuardrails:
    """Enhanced server-side guardrails with CRUD operation protection."""
    
    def __init__(self):
        # Critical patterns that could cause database harm
        self.dangerous_patterns = [
            re.compile(r'(?i)\$[a-z]+\s*:'),  # NoSQL operators
            re.compile(r'[;&|`]'),  # Command injection
            re.compile(r'\.\./'),  # Path traversal
        ]
        
        # CRUD operation patterns - REJECT THESE
        self.crud_operation_patterns = [
            # Write operations in queries
            re.compile(r'(?i)\$set\s*:'),
            re.compile(r'(?i)\$push\s*:'),
            re.compile(r'(?i)\$pull\s*:'),
            re.compile(r'(?i)\$unset\s*:'),
            re.compile(r'(?i)\$rename\s*:'),
            re.compile(r'(?i)\$inc\s*:'),
            re.compile(r'(?i)updateOne\s*\('),
            re.compile(r'(?i)updateMany\s*\('),
            re.compile(r'(?i)insertOne\s*\('),
            re.compile(r'(?i)insertMany\s*\('),
            re.compile(r'(?i)deleteOne\s*\('),
            re.compile(r'(?i)deleteMany\s*\('),
            re.compile(r'(?i)replaceOne\s*\('),
        ]

    def sanitize_parameters(self, carrier: str = "", flight_number: str = "", 
                          date_of_origin: str = "") -> Dict[str, Any]:
        """
        Enhanced parameter sanitization with CRUD protection.
        """
        # Basic sanitization - remove dangerous characters only
        safe_carrier = re.sub(r'[;&|`$]', '', carrier) if carrier else ""
        safe_flight_number = re.sub(r'[^0-9]', '', flight_number) if flight_number else ""
        safe_date = re.sub(r'[;&|`$]', '', date_of_origin) if date_of_origin else ""
        
        # Check for dangerous patterns in combined input
        combined = f"{safe_carrier}{safe_flight_number}{safe_date}"
        for pattern in self.dangerous_patterns:
            if pattern.search(combined):
                self._log_security_event("DANGEROUS_PATTERN", f"Pattern: {pattern.pattern}")
                return {"valid": False, "error": "Invalid parameters detected"}
        
        return {
            "valid": True, 
            "sanitized": {
                "carrier": safe_carrier,
                "flight_number": safe_flight_number,
                "date_of_origin": safe_date
            }
        }

    def validate_raw_query(self, query_json: str) -> Dict[str, Any]:
        """
        Enhanced raw query validation - block any CRUD operations.
        """
        # Check for CRUD operations in raw queries
        for pattern in self.crud_operation_patterns:
            if pattern.search(query_json):
                self._log_security_event("CRUD_OPERATION", f"Blocked CRUD in raw query: {pattern.pattern}")
                return {"valid": False, "error": "Modification operations are not allowed in raw queries"}
        
        # Check query complexity
        try:
            query_obj = json.loads(query_json)
            # Ensure it's a find query, not update/insert/delete
            if any(key.lower() in ['$set', '$push', '$pull', '$unset', '$inc'] for key in str(query_obj).split()):
                return {"valid": False, "error": "Modification operators are not permitted"}
        except:
            pass  # JSON parsing will be handled separately
        
        return {"valid": True, "message": "Raw query validation passed"}

    def _log_security_event(self, event_type: str, message: str):
        """Enhanced security logging."""
        logger.warning(f"SECURITY_BLOCKED: {event_type} - {message}")

# Initialize server guardrails
server_guardrails = ServerGuardrails()

# [ALL YOUR EXISTING SERVER FUNCTIONS REMAIN EXACTLY THE SAME]
# get_mongodb_client(), normalize_flight_number(), validate_date(), make_query()
# response_ok(), response_error(), _fetch_one_async()
# All remain unchanged

# --- MCP Tools with Enhanced CRUD Protection ---

@mcp.tool()
async def health_check() -> str:
    """Health check remains unchanged"""
    try:
        _, _, col = await get_mongodb_client()
        doc = await col.find_one({}, {"_id": 1})
        return response_ok({"status": "ok", "db_connected": doc is not None})
    except Exception as e:
        logger.exception("Health check DB ping failed")
        return response_error("DB unreachable", code=503)

@mcp.tool()
async def get_flight_basic_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """Enhanced with CRUD protection"""
    # Parameter validation with guardrails
    param_validation = server_guardrails.sanitize_parameters(carrier, flight_number, date_of_origin)
    if not param_validation["valid"]:
        return response_error(param_validation["error"], 400)
    
    # Use sanitized parameters
    safe_params = param_validation["sanitized"]
    carrier = safe_params["carrier"]
    flight_number = safe_params["flight_number"]
    date_of_origin = safe_params["date_of_origin"]
    
    # Your existing validation continues to work
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date_of_origin format. Expected YYYY-MM-DD or common date formats", 400)
    
    query = make_query(carrier, fn, dob)
    
    # Projection remains the same
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

# Apply the same CRUD protection pattern to all other tools...
# [Add parameter validation block to each tool function]

@mcp.tool()
async def raw_mongodb_query(query_json: str, limit: int = 10) -> str:
    """
    Run a raw MongoDB query string (JSON) against collection.
    Enhanced with CRUD operation blocking.
    """
    # Enhanced security for raw queries - BLOCK CRUD OPERATIONS
    if len(query_json) > 2000:
        return response_error("Query too long", 400)
    
    # Validate raw query for CRUD operations
    query_validation = server_guardrails.validate_raw_query(query_json)
    if not query_validation["valid"]:
        return response_error(query_validation["error"], 403)
    
    # Check for dangerous patterns
    for pattern in server_guardrails.dangerous_patterns:
        if pattern.search(query_json):
            server_guardrails._log_security_event("DANGEROUS_RAW_QUERY", f"Pattern: {pattern.pattern}")
            return response_error("Dangerous query pattern detected", 403)
    
    # Your existing code continues...
    try:
        _, _, col = await get_mongodb_client()
        try:
            query = json.loads(query_json)
            
            # Additional validation: ensure it's a find query, not update/insert/delete
            if not isinstance(query, dict):
                return response_error("Query must be a JSON object", 400)
                
            # Ensure no update operators are present
            query_str = json.dumps(query).lower()
            if any(op in query_str for op in ['$set', '$push', '$pull', '$unset', '$inc', '$rename']):
                return response_error("Modification operations are not permitted", 403)
                
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
