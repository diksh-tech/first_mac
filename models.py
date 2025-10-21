from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class ReturnEvent(BaseModel):
    returnEvent: str

class FlightData(BaseModel):
    flightlegstate: str
    carrier: str
    dateoforigin: str
    flight_number: str
    startstation: str
    endstation: str
    scheduledstarttime: datetime
    scheduledEndTime: datetime
    returnEvents: List[ReturnEvent]
