from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime
import uuid

class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    topic: str
    priority: int = Field(ge=1, le=10, default=5)
    is_anchor: bool = False  # True for the "First Message"
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict = {}