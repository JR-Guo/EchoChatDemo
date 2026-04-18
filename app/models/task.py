from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class TaskKind(str, Enum):
    REPORT = "report"
    MEASUREMENT = "measurement"
    DISEASE = "disease"
    VQA = "vqa"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    content: str
    edited: bool = False


class ReportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    sections: list[ReportSection] = []
    error: Optional[str] = None


class MeasurementItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    raw: str


class MeasurementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    items: list[MeasurementItem] = []
    error: Optional[str] = None


class DiseaseItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    answer: Literal["yes", "no", "unknown"]
    raw: str


class DiseaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    items: list[DiseaseItem] = []
    error: Optional[str] = None


class VQAMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class VQAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    messages: list[VQAMessage] = []
    error: Optional[str] = None
