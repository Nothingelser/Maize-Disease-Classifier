"""
Lightweight application records backed by Supabase row data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

def parse_datetime(value: Any) -> Optional[datetime]:
    """Convert ISO strings from Supabase into datetime objects when present."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None

@dataclass
class UserRecord:
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_admin: bool = False
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["UserRecord"]:
        if not data:
            return None
        return cls(
            id=data["id"],
            email=data.get("email", ""),
            username=data.get("username", ""),
            full_name=data.get("full_name"),
            avatar_url=data.get("avatar_url"),
            is_admin=bool(data.get("is_admin", False)),
            is_active=bool(data.get("is_active", True)),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            last_login=parse_datetime(data.get("last_login")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "is_admin": self.is_admin,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

@dataclass
class PredictionRecord:
    id: Any
    user_id: str
    image_name: str
    prediction: str
    confidence: float
    probabilities_json: Any = None
    processing_time: Optional[float] = None
    image_url: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    probabilities_cache: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["PredictionRecord"]:
        if not data:
            return None
        record = cls(
            id=data.get("id"),
            user_id=data["user_id"],
            image_name=data.get("image_name", ""),
            prediction=data.get("prediction", ""),
            confidence=float(data.get("confidence", 0)),
            probabilities_json=data.get("probabilities_json"),
            processing_time=data.get("processing_time"),
            image_url=data.get("image_url"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            created_at=parse_datetime(data.get("created_at")),
        )
        record.probabilities_cache = record.get_probabilities()
        return record

    def get_probabilities(self) -> List[Dict[str, Any]]:
        if self.probabilities_cache:
            return self.probabilities_cache
        if not self.probabilities_json:
            return []
        if isinstance(self.probabilities_json, list):
            return self.probabilities_json
        try:
            return json.loads(self.probabilities_json)
        except (TypeError, ValueError):
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_name": self.image_name,
            "image_url": self.image_url,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.get_probabilities(),
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
