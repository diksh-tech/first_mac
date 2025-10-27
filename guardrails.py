"""
Security Guardrails for FlightOps MCP System
Protects against injection, abuse, and malicious queries.
"""

import re
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger("FlightOps.Guardrails")

@dataclass
class ValidationResult:
    """Standardized validation result."""
    is_valid: bool
    reason: str = ""
    sanitized_input: Any = None
    error_code: int = 400

class SecurityGuardrails:
    """
    Comprehensive security guardrails for both client and server.
    """
    
    def __init__(self):
        # Request tracking for rate limiting (in production, use Redis)
        self.request_history = {}
        self.blocked_ips = set()
        
        # Compile regex patterns once for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all security regex patterns."""
        # Injection patterns
        self.injection_patterns = [
            re.compile(r'(?i)(drop\s+table|delete\s+from|insert\s+into|update\s+\w+\s+set)'),
            re.compile(r'(?i)(union\s+select|select\s+.+from|where\s+.+=)'),
            re.compile(r'(?i)(system\.|/etc/|/bin/|/usr/|\.\./)'),
            re.compile(r'(?i)(password|secret|api[_-]?key|token)\s*='),
            re.compile(r'[;&|`]\s*\w+'),
            re.compile(r'\$\s*\('),  # Command substitution
            re.compile(r'<\s*script'),  # XSS attempts
            re.compile(r'(?i)javascript:'),  # XSS attempts
        ]
        
        # Suspicious flight query patterns
        self.suspicious_query_patterns = [
            re.compile(r'(?i)(all\s+flights|every\s+flight|entire\s+database|show\s+me\s+everything)'),
            re.compile(r'(?i)(dump|export|download)\s+(data|flights|database)'),
            re.compile(r'(?i)flight\s+number\s*\>\s*\d{4}'),  # Range queries
            re.compile(r'(?i)carrier\s*=\s*["\']?.{4,}["\']?'),  # Long carrier codes
        ]
        
        # Valid patterns
        self.valid_carrier_pattern = re.compile(r'^[A-Z0-9]{2,3}$')
        self.valid_flight_number_pattern = re.compile(r'^\d{1,4}$')
        self.valid_date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

    # ==================== INPUT SANITIZATION ====================
    
    def sanitize_string(self, input_str: str, max_length: int = 100, 
                       allow_numbers: bool = True, allow_letters: bool = True) -> str:
        """
        Sanitize string input to prevent injection attacks.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            allow_numbers: Allow numeric characters
            allow_letters: Allow alphabetic characters
            
        Returns:
            Sanitized string
        """
        if not input_str:
            return ""
        
        # Remove potentially dangerous characters
        safe_chars = ""
        if allow_letters:
            safe_chars += r'A-Za-z'
        if allow_numbers:
            safe_chars += r'0-9'
        
        # Allow basic punctuation for flight queries
        safe_chars += r'\-_/. '
        
        pattern = f'[^{safe_chars}]'
        sanitized = re.sub(pattern, '', str(input_str))
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()

    def sanitize_flight_parameters(self, carrier: str = "", flight_number: str = "", 
                                 date_of_origin: str = "") -> Tuple[str, str, str]:
        """
        Sanitize all flight-related parameters.
        
        Returns:
            Tuple of (sanitized_carrier, sanitized_flight_number, sanitized_date)
        """
        safe_carrier = self.sanitize_string(carrier, max_length=3, allow_numbers=True, allow_letters=True)
        safe_flight_number = self.sanitize_string(flight_number, max_length=4, allow_numbers=True, allow_letters=False)
        safe_date = self.sanitize_string(date_of_origin, max_length=10, allow_numbers=True, allow_letters=False)
        
        return safe_carrier, safe_flight_number, safe_date

    # ==================== VALIDATION METHODS ====================
    
    def validate_carrier_code(self, carrier: str) -> ValidationResult:
        """Validate airline carrier code format."""
        if not carrier:
            return ValidationResult(True, "Empty carrier allowed", "")
        
        if not self.valid_carrier_pattern.match(carrier):
            return ValidationResult(
                False, 
                f"Invalid carrier code: {carrier}. Must be 2-3 alphanumeric characters.",
                carrier,
                400
            )
        
        return ValidationResult(True, "Valid carrier code", carrier)

    def validate_flight_number(self, flight_number: str) -> ValidationResult:
        """Validate flight number format and range."""
        if not flight_number:
            return ValidationResult(True, "Empty flight number allowed", "")
        
        # Check format
        if not self.valid_flight_number_pattern.match(flight_number):
            return ValidationResult(
                False,
                f"Invalid flight number: {flight_number}. Must be 1-4 digits.",
                flight_number,
                400
            )
        
        # Check realistic range
        fn_int = int(flight_number)
        if not (1 <= fn_int <= 9999):
            return ValidationResult(
                False,
                f"Flight number out of range: {flight_number}. Must be between 1-9999.",
                flight_number,
                400
            )
        
        return ValidationResult(True, "Valid flight number", flight_number)

    def validate_date(self, date_str: str) -> ValidationResult:
        """Validate date format and logical range."""
        if not date_str:
            return ValidationResult(True, "Empty date allowed", "")
        
        # Basic format check
        if not self.valid_date_pattern.match(date_str):
            return ValidationResult(
                False,
                f"Invalid date format: {date_str}. Must be YYYY-MM-DD.",
                date_str,
                400
            )
        
        # Logical date range (2000-2030)
        try:
            year = int(date_str[:4])
            if not (2000 <= year <= 2030):
                return ValidationResult(
                    False,
                    f"Date year out of range: {year}. Must be between 2000-2030.",
                    date_str,
                    400
                )
        except ValueError:
            return ValidationResult(
                False,
                f"Invalid date: {date_str}",
                date_str,
                400
            )
        
        return ValidationResult(True, "Valid date", date_str)

    def validate_query_scope(self, carrier: str, flight_number: str, date_of_origin: str) -> ValidationResult:
        """
        Validate that query isn't too broad for safety.
        """
        # If all parameters are empty, query is too broad
        if not carrier and not flight_number and not date_of_origin:
            return ValidationResult(
                False,
                "Query too broad. Please specify at least one of: carrier, flight_number, or date_of_origin.",
                None,
                400
            )
        
        # If only date is provided, might be too many results
        if date_of_origin and not carrier and not flight_number:
            return ValidationResult(
                False,
                "Query with only date is too broad. Please also specify carrier or flight_number.",
                None,
                400
            )
        
        return ValidationResult(True, "Query scope acceptable")

    # ==================== SECURITY CHECKS ====================
    
    def detect_injection_attempt(self, text: str) -> ValidationResult:
        """Detect potential injection attacks in any text input."""
        if not text:
            return ValidationResult(True, "No text to check")
        
        text_lower = text.lower()
        
        for pattern in self.injection_patterns:
            if pattern.search(text_lower):
                return ValidationResult(
                    False,
                    f"Potential injection attempt detected: {text}",
                    text,
                    403
                )
        
        return ValidationResult(True, "No injection attempts detected")

    def detect_suspicious_query(self, query: str) -> ValidationResult:
        """Detect suspicious or overly broad queries."""
        if not query:
            return ValidationResult(True, "No query to check")
        
        query_lower = query.lower()
        
        for pattern in self.suspicious_query_patterns:
            if pattern.search(query_lower):
                return ValidationResult(
                    False,
                    f"Suspicious query pattern detected: {query}",
                    query,
                    403
                )
        
        # Check query length
        if len(query) > 500:
            return ValidationResult(
                False,
                f"Query too long: {len(query)} characters. Maximum 500 allowed.",
                query,
                400
            )
        
        return ValidationResult(True, "Query appears safe")

    # ==================== RATE LIMITING ====================
    
    def check_rate_limit(self, client_ip: str, max_requests: int = 60, window_seconds: int = 60) -> ValidationResult:
        """
        Basic in-memory rate limiting.
        In production, replace with Redis.
        """
        if client_ip in self.blocked_ips:
            return ValidationResult(
                False,
                "IP address temporarily blocked due to excessive requests",
                None,
                429
            )
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Clean old entries
        if client_ip in self.request_history:
            self.request_history[client_ip] = [
                t for t in self.request_history[client_ip] 
                if t > window_start
            ]
        else:
            self.request_history[client_ip] = []
        
        # Check rate limit
        if len(self.request_history[client_ip]) >= max_requests:
            # Block for 5 minutes if rate limit exceeded
            self.blocked_ips.add(client_ip)
            # Schedule unblock (in production, use proper TTL)
            asyncio.create_task(self._unblock_ip_after_delay(client_ip, 300))
            
            return ValidationResult(
                False,
                f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds.",
                None,
                429
            )
        
        # Record this request
        self.request_history[client_ip].append(current_time)
        return ValidationResult(True, "Rate limit OK")

    async def _unblock_ip_after_delay(self, ip: str, delay_seconds: int):
        """Unblock IP after delay."""
        await asyncio.sleep(delay_seconds)
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked IP: {ip}")

    # ==================== COMPREHENSIVE VALIDATION ====================
    
    def validate_flight_query(self, carrier: str = "", flight_number: str = "", 
                            date_of_origin: str = "", client_ip: str = "unknown") -> ValidationResult:
        """
        Comprehensive validation for flight queries.
        
        Returns:
            ValidationResult with overall validation status
        """
        # Step 1: Rate limiting
        rate_check = self.check_rate_limit(client_ip)
        if not rate_check.is_valid:
            return rate_check
        
        # Step 2: Sanitize inputs
        safe_carrier, safe_flight_number, safe_date = self.sanitize_flight_parameters(
            carrier, flight_number, date_of_origin
        )
        
        # Step 3: Validate individual parameters
        carrier_validation = self.validate_carrier_code(safe_carrier)
        if not carrier_validation.is_valid:
            return carrier_validation
        
        flight_validation = self.validate_flight_number(safe_flight_number)
        if not flight_validation.is_valid:
            return flight_validation
        
        date_validation = self.validate_date(safe_date)
        if not date_validation.is_valid:
            return date_validation
        
        # Step 4: Check query scope
        scope_validation = self.validate_query_scope(safe_carrier, safe_flight_number, safe_date)
        if not scope_validation.is_valid:
            return scope_validation
        
        # Step 5: Check for injection in combined parameters
        combined_text = f"{safe_carrier} {safe_flight_number} {safe_date}"
        injection_check = self.detect_injection_attempt(combined_text)
        if not injection_check.is_valid:
            return injection_check
        
        return ValidationResult(
            True, 
            "All validations passed",
            {
                "carrier": safe_carrier,
                "flight_number": safe_flight_number,
                "date_of_origin": safe_date
            }
        )

    def validate_user_message(self, message: str, client_ip: str = "unknown") -> ValidationResult:
        """
        Validate user message for security before LLM processing.
        """
        # Rate limiting
        rate_check = self.check_rate_limit(client_ip)
        if not rate_check.is_valid:
            return rate_check
        
        # Message security checks
        injection_check = self.detect_injection_attempt(message)
        if not injection_check.is_valid:
            return injection_check
        
        suspicious_check = self.detect_suspicious_query(message)
        if not suspicious_check.is_valid:
            return suspicious_check
        
        return ValidationResult(True, "User message validation passed", message)

    # ==================== SECURITY LOGGING ====================
    
    def log_security_event(self, event_type: str, message: str, user_input: str = "", 
                         client_ip: str = "unknown", severity: str = "INFO"):
        """Enhanced security logging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "user_input": user_input[:200],  # Truncate for safety
            "client_ip": client_ip,
            "component": "SecurityGuardrails"
        }
        
        log_message = json.dumps(log_entry)
        
        if severity == "ERROR":
            logger.error(log_message)
        elif severity == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # In production, also send to security monitoring system
        return log_entry

# ==================== CLIENT-SIDE GUARDRAILS ====================

def get_client_guardrails_prompt(tools_prompt: str) -> str:
    """
    Generate comprehensive client-side guardrails system prompt.
    """
    return f"""
You are an assistant that converts user questions into MCP tool calls with STRICT SECURITY GUARDRAILS.

Available tools:
{tools_prompt}

## MANDATORY SECURITY GUARDRAILS:

1. **INPUT VALIDATION**:
   - Sanitize all user inputs before tool calls
   - Remove special characters: < > & ; | $ ` ( ) {{ }}
   - Trim whitespace from all parameters
   - Validate flight numbers are numeric (1-9999)
   - Validate dates are in YYYY-MM-DD format

2. **QUERY SAFETY FILTERS**:
   - REJECT queries requesting: database modifications, system commands, file operations
   - REJECT queries for sensitive data beyond flight information
   - REJECT overly broad queries (e.g., "all flights", "show me everything")
   - REJECT multiple unrelated flights in single query
   - If query seems malicious, return: {{"plan": [], "error": "Query rejected by security filters"}}

3. **TOOL USAGE RULES**:
   - Use ONLY the tools listed above - no exceptions
   - Tool names must match EXACTLY (case-sensitive)
   - Maximum 3 tool calls per query
   - Prefer specific queries over broad searches

4. **PARAMETER HANDLING**:
   - For flight numbers: must be numeric (1-9999)
   - For dates: must be valid format (YYYY-MM-DD)
   - For carriers: 2-3 character airline codes only
   - If parameters missing/invalid, omit them rather than using "unknown"
   - Never include "unknown", "any", or placeholder values

5. **SECURITY RESPONSES**:
   - If query is rejected, provide clear security reason
   - Never reveal internal system details in errors
   - Log security events for monitoring

## SUSPICIOUS PATTERNS TO REJECT:
- Database operations (DROP, INSERT, UPDATE, DELETE)
- System commands (/, etc, bin, system.)
- Sensitive data requests (passwords, keys, tokens)
- Overly broad queries ("all flights", "everything")
- Multiple flight numbers in one query
- Special characters in parameters

## OUTPUT FORMAT:
{{
  "plan": [
    {{
      "tool": "tool_name",
      "arguments": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }}
  ]
}}

## SAFETY FIRST - REJECT ANY SUSPICIOUS QUERIES IMMEDIATELY.
"""

# Global instance
security_guardrails = SecurityGuardrails()

# Utility functions for easy import
def validate_query(carrier: str = "", flight_number: str = "", date_of_origin: str = "", 
                  client_ip: str = "unknown") -> ValidationResult:
    """Convenience function for query validation."""
    return security_guardrails.validate_flight_query(carrier, flight_number, date_of_origin, client_ip)

def validate_user_input(message: str, client_ip: str = "unknown") -> ValidationResult:
    """Convenience function for user input validation."""
    return security_guardrails.validate_user_message(message, client_ip)

def log_security_alert(event_type: str, message: str, user_input: str = "", 
                      client_ip: str = "unknown", severity: str = "WARNING"):
    """Convenience function for security logging."""
    return security_guardrails.log_security_event(event_type, message, user_input, client_ip, severity)
    ########################################################################################################
from guardrails import validate_query, log_security_alert

@mcp.tool()
async def get_flight_basic_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    # Validate inputs first
    validation = validate_query(carrier, flight_number, date_of_origin, client_ip="127.0.0.1")
    if not validation.is_valid:
        log_security_alert("INVALID_QUERY", validation.reason, f"carrier={carrier}", "127.0.0.1", "WARNING")
        return response_error(validation.reason, validation.error_code)
    
    # Use sanitized parameters
    safe_params = validation.sanitized_input
    # ... rest of your code
  ############################################################################################################
from guardrails import get_client_guardrails_prompt, validate_user_input

# Replace your existing SYSTEM_PROMPT_PLAN
SYSTEM_PROMPT_PLAN = get_client_guardrails_prompt(_build_tool_prompt())

# In run_query method:
async def run_query(self, user_query: str) -> dict:
    # Validate user input first
    validation = validate_user_input(user_query, client_ip="client")
    if not validation.is_valid:
        return {"error": f"Security rejection: {validation.reason}"}
    
    # Continue with existing logic...
  ####################################################################################################
