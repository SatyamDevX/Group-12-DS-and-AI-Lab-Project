"""Compatibility wrapper for the old translator2 module."""
from app.models.translator import load_llm, load_tokenizer, translate


def generate_translation(model, tokenizer, hindi_text: str) -> str:
    return translate(hindi_text, model, tokenizer)
