"""System prompts and query templates per task.

Source of truth: docs/{report,measurement,disease}(1).py.
VQA had no prior prompt; we pick a bounded system prompt here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    query_template: str  # may contain substitution tokens described per-task


REPORT_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Your task is to write a sectioned "
        "echo report including sections of: Aortic Valve, Atria, Great Vessels, "
        "Left Ventricle, Mitral Valve, Pericardium Pleural, Pulmonic Valve, "
        "Right Ventricle, Tricuspid Valve, and Summary, from the given "
        "echocardiography."
    ),
    query_template="Write a report from the echocardiography.",
)

MEASUREMENT_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Your task is to measure heart "
        "parameters from the given echocardiography."
    ),
    query_template="Please measure the <measure>.",
)

DISEASE_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Your task is to diagnose heart "
        "conditions from the given echocardiography. Answer yes or no."
    ),
    query_template="Based on the echocardiography, does the patient have <disease>?",
)

VQA_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Answer the user's question about "
        "the provided echocardiography concisely and clinically. If the "
        "question cannot be answered from the given images, state that "
        "explicitly. Do not answer non-echocardiography questions."
    ),
    query_template="{question}",
)
