"""Pydantic schemas for Study, Clip, and TasksAvailability.

Study represents a single upload session; Clip represents one uploaded file
(dicom or image/video). TasksAvailability summarizes which downstream tasks
can be run based on the set of view groups present in the clips.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ClipKind = Literal["dicom", "image", "video"]


class Clip(BaseModel):
    """A single uploaded clip or image."""

    file_id: str
    original_filename: str
    kind: ClipKind
    raw_path: str
    converted_path: Optional[str] = None
    view: Optional[str] = None            # model-predicted view label
    user_view: Optional[str] = None       # user-overridden view label
    confidence: Optional[float] = None    # classifier confidence in [0, 1]
    is_video: bool = False


class TasksAvailability(BaseModel):
    """Per-task runnability flags derived from clip view coverage."""

    report: bool
    measurement: bool
    disease: bool
    vqa: bool
    missing_groups: list[str] = Field(default_factory=list)


class Study(BaseModel):
    """A single upload session grouping one or more clips."""

    study_id: str
    created_at: datetime
    clips: list[Clip] = Field(default_factory=list)
    tasks: TasksAvailability
