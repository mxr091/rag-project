"""Model configuration is read only when model generation is explicitly enabled."""
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .parser import PdfError


def configured_model():
    endpoint = os.getenv("MODEL_ENDPOINT", "")
    parsed = urlsplit(endpoint)
    key, name = os.getenv("MODEL_API_KEY"), os.getenv("MODEL_NAME")
    if (not key or not name or parsed.scheme != "https" or not parsed.hostname
            or parsed.username or parsed.password):
        raise PdfError("model_not_configured")
    adapter = Path(__file__).resolve().parents[3] / "experiments" / "02-openai-compatible-adapter"
    if str(adapter) not in sys.path:
        sys.path.insert(0, str(adapter))
    from openai_compatible import OpenAICompatibleModel
    return OpenAICompatibleModel(endpoint=endpoint, api_key=key, model=name,
        timeout_seconds=45, max_retries=0, max_tokens=1200,
        thinking_type=os.getenv("MODEL_THINKING_TYPE") or ("disabled" if parsed.hostname == "api.deepseek.com" else None))
