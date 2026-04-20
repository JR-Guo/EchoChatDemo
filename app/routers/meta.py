from fastapi import APIRouter

from constants.diseases import SUPPORT_DISEASES
from constants.measurements import SUPPORT_MEASUREMENTS
from constants.report_sections import REPORT_SECTIONS
from constants.view_labels import VIEW_LABELS
from constants.presets import DISEASE_PRESETS, MEASUREMENT_PRESETS, VQA_EXAMPLES

router = APIRouter()


@router.get("/api/constants")
def constants():
    return {
        "diseases": SUPPORT_DISEASES,
        "measurements": SUPPORT_MEASUREMENTS,
        "report_sections": REPORT_SECTIONS,
        "views": VIEW_LABELS,
        "presets": {
            "measurements": MEASUREMENT_PRESETS,
            "diseases": DISEASE_PRESETS,
            "vqa_examples": VQA_EXAMPLES,
        },
    }
