"""
Shared Gemini call helper, adapted from job-auto-applier/tailoring/gemini_client.py
(same free-tier fallback-chain pattern, proven live in that project).

Uses Gemini's free tier (ai.google.dev) rather than a paid API -- each model
has its own separate daily quota bucket, so generate() walks a priority
chain of models on RESOURCE_EXHAUSTED rather than failing the moment the
top choice is exhausted, and always re-tries the best model first on the
next call (no cached cooldown state) so it recovers the instant a quota
resets.
"""

import os

from core import _net  # noqa: F401 -- must run before any network call, see its docstring
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_MODEL = "gemini-3.5-flash"
MODEL_CHAIN = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
]

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY not set -- copy .env.example to .env and add a free key "
                "from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _call_model(model: str, system: str, user: str, max_output_tokens: int) -> str:
    response = _get_client().models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


def generate(system: str, user: str, max_output_tokens: int = 2048) -> str:
    last_exc: Exception | None = None
    for candidate in MODEL_CHAIN:
        try:
            return _call_model(candidate, system, user, max_output_tokens)
        except ClientError as exc:
            # RESOURCE_EXHAUSTED = this model's free-tier quota is used up;
            # NOT_FOUND = this model name has been deprecated/retired.
            # INVALID_ARGUMENT is normally a real bad-request bug worth
            # surfacing immediately -- but found live: a real 400
            # INVALID_ARGUMENT on one specific (likely preview/experimental)
            # model in the chain did NOT reproduce seconds later against the
            # exact same prompt, so it's model-specific flakiness at least
            # some of the time, not a deterministic malformed request. Worth
            # one pass through the rest of the chain before giving up --
            # if every model rejects the same prompt, that's real signal;
            # if only one does, this recovers automatically.
            if getattr(exc, "status", None) in ("RESOURCE_EXHAUSTED", "NOT_FOUND", "INVALID_ARGUMENT"):
                last_exc = exc
                continue
            raise
        except ServerError as exc:
            # transient overload (503), not a quota issue -- worth trying
            # the next model rather than failing the whole analysis on one
            # model's momentary bad luck.
            last_exc = exc
            continue
    raise last_exc
