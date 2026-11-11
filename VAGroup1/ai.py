"""
ai.py - Root re-export for app/ai.py
Allows importing either via 'import ai' or 'import app.ai'.
"""
from app.ai import (
    INSUFFICIENT_DATA_MESSAGE,
    BANNED_FINANCIAL_WORDS,
    is_low_confidence,
    generate_template_narrative,
    extract_numbers_from_text,
    extract_allowed_numbers_from_dict,
    is_number_allowed,
    verify_narrative,
    generate_insight_narrative,
)

__all__ = [
    "INSUFFICIENT_DATA_MESSAGE",
    "BANNED_FINANCIAL_WORDS",
    "is_low_confidence",
    "generate_template_narrative",
    "extract_numbers_from_text",
    "extract_allowed_numbers_from_dict",
    "is_number_allowed",
    "verify_narrative",
    "generate_insight_narrative",
]

