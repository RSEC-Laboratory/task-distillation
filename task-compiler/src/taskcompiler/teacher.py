"""Call the teacher model and track its usage."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import anthropic
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from taskcompiler.config import (
    ANTHROPIC_TEACHER,
    OPENROUTER_TEACHER,
    PROJECT_ROOT,
    TEACHER_BACKENDS,
)


@dataclass
class TeacherResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    model: str
    backend: str = ANTHROPIC_TEACHER
    finish_reason: str | None = None


class TeacherClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        backend: str = ANTHROPIC_TEACHER,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 180.0,
        provider: str | None = None,
        max_retries: int | None = None,
    ):
        if backend not in TEACHER_BACKENDS:
            raise ValueError(f"unknown teacher backend: {backend}")
        if timeout_seconds <= 0:
            raise ValueError("teacher timeout must be positive")
        if max_retries is not None and (type(max_retries) is not int or max_retries < 0):
            raise ValueError("max_retries must be a nonnegative integer")
        if not api_key:
            load_dotenv(PROJECT_ROOT / ".env")
        key_name = (
            "OPENROUTER_API_KEY"
            if backend == OPENROUTER_TEACHER
            else "ANTHROPIC_API_KEY"
        )
        key = api_key or os.environ.get(key_name)
        if not key:
            raise RuntimeError(
                f"{key_name} is not set in the environment or project .env file."
            )
        self.model = model
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")
        self.provider = provider.strip() if provider else None
        self.identity = {
            "backend": backend,
            "requested_model": model,
            "base_url": self.base_url if backend == OPENROUTER_TEACHER else None,
            "temperature": 0,
            "provider": self.provider,
            "max_retries": max_retries,
        }
        if backend == ANTHROPIC_TEACHER:
            options = {} if max_retries is None else {"max_retries": max_retries}
            self._client = anthropic.Anthropic(api_key=key, **options)
            return

        self._client = requests.Session()
        retries = Retry(
            total=5 if max_retries is None else max_retries,
            connect=5 if max_retries is None else max_retries,
            read=2 if max_retries is None else max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"POST"}),
        )
        adapter = HTTPAdapter(max_retries=retries, pool_maxsize=4)
        self._client.mount("https://", adapter)
        self._client.mount("http://", adapter)
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def call(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 2000,
        response_format: dict | None = None,
        seed: int | None = None,
    ) -> TeacherResponse:
        if self.backend == OPENROUTER_TEACHER:
            return self._call_openrouter(
                system,
                prompt,
                max_tokens,
                response_format,
                seed,
            )

        start = time.monotonic()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.monotonic() - start
        text = "".join(block.text for block in message.content if block.type == "text")
        return TeacherResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_seconds=elapsed,
            model=self.model,
            backend=self.backend,
            finish_reason=getattr(message, "stop_reason", None),
        )

    def _call_openrouter(
        self,
        system: str,
        prompt: str,
        max_tokens: int,
        response_format: dict | None,
        seed: int | None,
    ) -> TeacherResponse:
        start = time.monotonic()
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "reasoning": {"enabled": False},
        }
        if self.provider:
            request_body["provider"] = {
                "order": [self.provider],
                "allow_fallbacks": False,
            }
        if response_format is not None:
            request_body["response_format"] = response_format
        if seed is not None:
            request_body["seed"] = seed
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=request_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                detail = exc.response.text.strip().replace("\n", " ")[:500]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"OpenRouter request failed: {exc}{suffix}"
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        elapsed = time.monotonic() - start

        try:
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an invalid chat response") from exc
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter returned no final answer text")
        resolved_model = str(payload.get("model") or self.model)
        finish_reason = payload["choices"][0].get("finish_reason")
        self.identity["resolved_model"] = resolved_model
        self.identity["provider"] = payload.get("provider")
        return TeacherResponse(
            text=content.strip(),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_seconds=elapsed,
            model=resolved_model,
            backend=self.backend,
            finish_reason=finish_reason,
        )
