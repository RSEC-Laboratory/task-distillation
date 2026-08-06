"""Run frozen-student inference through a configured model backend."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests
import torch
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib3.util.retry import Retry

from taskcompiler.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    OLLAMA_BACKEND,
    OLLAMA_OPENROUTER_BACKEND,
    OPENROUTER_BACKEND,
    PROJECT_ROOT,
    TRANSFORMERS_BACKEND,
)


@dataclass
class StudentResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    finish_reason: str | None = None


class StudentModel:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_new_tokens: int = 400,
        revision: str | None = None,
        backend: str = TRANSFORMERS_BACKEND,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_timeout_seconds: float = 120.0,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_timeout_seconds: float = 180.0,
        openrouter_model_name: str | None = None,
        openrouter_provider: str | None = None,
        context_tokens: int = 32768,
        enable_thinking: bool = False,
        seed: int = 0,
    ):
        if backend not in {
            TRANSFORMERS_BACKEND,
            OLLAMA_BACKEND,
            OPENROUTER_BACKEND,
            OLLAMA_OPENROUTER_BACKEND,
        }:
            raise ValueError(f"unknown student backend: {backend}")
        if backend == OPENROUTER_BACKEND and "/" not in model_name:
            raise ValueError(
                "OpenRouter student models require a provider/model identifier"
            )
        if (
            backend == OLLAMA_OPENROUTER_BACKEND
            and (
                not openrouter_model_name
                or "/" not in openrouter_model_name
            )
        ):
            raise ValueError(
                "Ollama fallback requires an OpenRouter provider/model identifier"
            )
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if ollama_timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive")
        if openrouter_timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive")
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")

        self.model_name = model_name
        self.ollama_model_name = model_name
        self.openrouter_model_name = openrouter_model_name
        self.max_new_tokens = max_new_tokens
        self.revision = revision
        self.requested_backend = backend
        self.backend = backend
        self.ollama_base_url = (
            _normalize_ollama_url(ollama_base_url)
            if backend in {
                OLLAMA_BACKEND,
                OLLAMA_OPENROUTER_BACKEND,
            }
            else ollama_base_url
        )
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self.openrouter_base_url = openrouter_base_url.rstrip("/")
        self.openrouter_timeout_seconds = openrouter_timeout_seconds
        self.openrouter_provider = (
            openrouter_provider.strip() if openrouter_provider else None
        )
        self.context_tokens = context_tokens
        self.enable_thinking = enable_thinking
        self.seed = seed

        if backend in {
            OLLAMA_BACKEND,
            OPENROUTER_BACKEND,
            OLLAMA_OPENROUTER_BACKEND,
        }:
            self._session = requests.Session()
            retries = Retry(
                total=5,
                connect=5,
                read=0,
                backoff_factor=0.25,
                status_forcelist=(429, 502, 503, 504),
                allowed_methods=frozenset({"GET", "POST"}),
            )
            self._session.mount(
                "http://", HTTPAdapter(max_retries=retries, pool_maxsize=4)
            )
            self._session.mount(
                "https://", HTTPAdapter(max_retries=retries, pool_maxsize=4)
            )
            self.device = None
            self.tokenizer = None
            self.model = None
            if backend in {
                OLLAMA_BACKEND,
                OLLAMA_OPENROUTER_BACKEND,
            }:
                try:
                    digest = self._resolve_ollama_digest()
                except RuntimeError as exc:
                    if backend != OLLAMA_OPENROUTER_BACKEND:
                        raise
                    self._activate_openrouter(
                        openrouter_model_name,
                        fallback_reason=str(exc),
                    )
                    return
                self.backend = OLLAMA_BACKEND
                self.identity = {
                    "requested_backend": backend,
                    "backend": OLLAMA_BACKEND,
                    "model": model_name,
                    "digest": digest,
                    "base_url": self.ollama_base_url,
                    "fallback_used": False,
                    "seed": seed,
                    "temperature": 0,
                    "context_tokens": context_tokens,
                    "thinking": enable_thinking,
                }
            else:
                self._activate_openrouter(model_name)
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.identity = {
            "backend": backend,
            "model": model_name,
            "revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "context_tokens": context_tokens,
            "thinking": enable_thinking,
        }

    def _activate_openrouter(
        self,
        model_name: str | None,
        fallback_reason: str | None = None,
    ) -> None:
        """Select OpenRouter before the experiment starts."""
        if not model_name:
            raise RuntimeError("OpenRouter fallback model is not configured")
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set in the environment "
                "or project .env file."
            )
        self.backend = OPENROUTER_BACKEND
        self.model_name = model_name
        self._openrouter_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.identity = {
            "requested_backend": self.requested_backend,
            "backend": OPENROUTER_BACKEND,
            "requested_model": model_name,
            "base_url": self.openrouter_base_url,
            "provider": self.openrouter_provider,
            "fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "ollama_model": self.ollama_model_name,
            "ollama_base_url": self.ollama_base_url,
            "seed": self.seed,
            "temperature": 0,
            "requested_context_tokens": self.context_tokens,
            "thinking": self.enable_thinking,
        }

    def ask(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: dict | None = None,
    ) -> StudentResponse:
        if self.backend == OLLAMA_BACKEND:
            return self._ask_ollama(prompt, system, response_schema)
        if self.backend == OPENROUTER_BACKEND:
            return self._ask_openrouter(prompt, system, response_schema)
        if response_schema is not None:
            raise ValueError(
                "structured response schemas require Ollama or OpenRouter"
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        input_token_count = input_ids.shape[1]

        start = time.monotonic()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - start

        new_tokens = output_ids[0][input_token_count:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return StudentResponse(
            text=text.strip(),
            input_tokens=input_token_count,
            output_tokens=new_tokens.shape[0],
            latency_seconds=elapsed,
        )

    def _ask_ollama(
        self,
        prompt: str,
        system: str | None,
        response_schema: dict | None,
    ) -> StudentResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            request = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "think": self.enable_thinking,
                "options": {
                    "temperature": 0,
                    "seed": self.seed,
                    "num_predict": self.max_new_tokens,
                    "num_ctx": self.context_tokens,
                },
            }
            if response_schema is not None:
                request["format"] = response_schema
            response = self._session.post(
                f"{self.ollama_base_url}/api/chat",
                json=request,
                timeout=self.ollama_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"Ollama request failed at {self.ollama_base_url}: {exc}"
            ) from exc
        elapsed = time.monotonic() - start

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an invalid chat response")
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama returned an invalid chat response")
        return StudentResponse(
            text=message["content"].strip(),
            input_tokens=int(payload.get("prompt_eval_count") or 0),
            output_tokens=int(payload.get("eval_count") or 0),
            latency_seconds=elapsed,
            finish_reason=(
                str(payload["done_reason"])
                if payload.get("done_reason") is not None
                else None
            ),
        )

    def _resolve_ollama_digest(self) -> str:
        try:
            response = self._session.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=min(self.ollama_timeout_seconds, 10.0),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"Could not identify the Ollama model at {self.ollama_base_url}: {exc}"
            ) from exc

        expected_name = (
            self.model_name if ":" in self.model_name else f"{self.model_name}:latest"
        )
        for model in payload.get("models", []):
            if model.get("name") == expected_name and model.get("digest"):
                return str(model["digest"])
        raise RuntimeError(
            f"Ollama model {self.model_name!r} is not installed at "
            f"{self.ollama_base_url}"
        )

    def _ask_openrouter(
        self,
        prompt: str,
        system: str | None,
        response_schema: dict | None,
    ) -> StudentResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": self.max_new_tokens,
            "reasoning": {"enabled": self.enable_thinking},
        }
        if self.openrouter_provider:
            request["provider"] = {
                "order": [self.openrouter_provider],
                "allow_fallbacks": False,
            }
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "student_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        start = time.monotonic()
        try:
            response = self._session.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=self._openrouter_headers,
                json=request,
                timeout=self.openrouter_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                detail = exc.response.text.strip().replace("\n", " ")[:500]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"OpenRouter student request failed: {exc}{suffix}"
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"OpenRouter student request failed: {exc}"
            ) from exc
        elapsed = time.monotonic() - start
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            usage = payload.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenRouter returned an invalid student response"
            ) from exc
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "OpenRouter returned no student response text"
            )
        self.identity["resolved_model"] = str(
            payload.get("model") or self.model_name
        )
        self.identity["provider"] = payload.get("provider")
        return StudentResponse(
            text=content.strip(),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_seconds=elapsed,
            finish_reason=(
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            ),
        )


def _normalize_ollama_url(base_url: str) -> str:
    """Normalize an Ollama server URL."""
    value = base_url.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Ollama URL must use http or https")
    if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must contain only a scheme, host, and port")
    if parsed.port is None:
        value = f"{value}:11434"
    return value
