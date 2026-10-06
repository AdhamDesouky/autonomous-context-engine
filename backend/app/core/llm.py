from __future__ import annotations

import os
from typing import Literal

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

Provider = Literal["google", "groq"]


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required for the configured LLM provider. "
            f"Set it in backend/.env or the process environment."
        )
    return value


def _google_api_keys() -> list[str]:
    primary = _required_setting("GOOGLE_API_KEY")
    backups = [
        key.strip()
        for key in os.getenv("GOOGLE_API_KEYS", "").split(",")
        if key.strip()
    ]
    return list(dict.fromkeys([primary, *backups]))


def create_chat_model(provider: Provider, *, role: str) -> Runnable:
    if provider == "google":
        model_name = os.getenv("GOOGLE_MODEL", "gemini-3.7-flash")
        models = [
            ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model_name,
                temperature=0,
                max_retries=0,
            )
            for api_key in _google_api_keys()
        ]
        return models[0].with_fallbacks(
            models[1:],
            exceptions_to_handle=(Exception,),
        )

    if provider == "groq":
        return ChatOpenAI(
            api_key=_required_setting("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0,
            max_retries=2,
        )

    raise ValueError(f"Unsupported {role} provider: {provider}")


def configured_provider(role: str) -> Provider:
    setting_name = f"{role.upper()}_PROVIDER"
    value = os.getenv(setting_name, "google").strip().lower()
    if value not in {"google", "groq"}:
        raise ValueError(
            f"{setting_name} must be either 'google' or 'groq', got {value!r}"
        )
    return value  # type: ignore[return-value]
