"""Small, dependency-free adapter for an explicitly private Ollama endpoint."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.request
from urllib.parse import urlsplit

from .experiment import ModelCallError, Response


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Model endpoint redirects are disabled")


def validate_private_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Use an HTTP(S) private endpoint without credentials in its URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must be an origin without path, query, or fragment")
    host = parsed.hostname
    _ = parsed.port  # urllib validates the port range here, before any request.
    if host != "localhost":
        try:
            address = ipaddress.ip_address(host or "")
        except ValueError as exc:
            raise ValueError("Use localhost or an explicit private/loopback IP address") from exc
        private_networks = (ipaddress.ip_network("10.0.0.0/8"),
                            ipaddress.ip_network("172.16.0.0/12"),
                            ipaddress.ip_network("192.168.0.0/16"),
                            ipaddress.ip_network("fc00::/7"))
        if not address.is_loopback and not any(address in network for network in private_networks):
            raise ValueError("Remote/public student endpoints are forbidden by this protocol")
    return url.rstrip("/")


class OllamaLocal:
    """No hosted fallback, download, credential forwarding, or HTTP proxy use.

    Private-network endpoints still rely on the user's control of that server.
    This boundary is an access policy, not a formal privacy guarantee.
    """

    def __init__(self, model: str, *, endpoint: str = "http://127.0.0.1:11434",
                 seed: int = 0, max_output_tokens: int = 512,
                 context_tokens: int = 8192, timeout: float = 120):
        self.endpoint = validate_private_endpoint(endpoint)
        if max_output_tokens < 1 or context_tokens < 1 or timeout <= 0:
            raise ValueError("Token limits and timeout must be positive")
        if "cloud" in model.casefold():
            raise ValueError("Cloud-routed Ollama models are not permitted")
        self.model = model
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.context_tokens = context_tokens
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        tags = self._request("/api/tags")
        matches = [m for m in tags.get("models", [])
                   if m.get("name") in {model, model + ":latest"}]
        if len(matches) != 1 or not matches[0].get("digest"):
            raise ValueError("Requested model must already exist locally with a digest")
        info = self._request("/api/show", {"model": model})
        if any(info.get(key) for key in ("remote_model", "remote_host")):
            raise ValueError("Ollama reports a remote/cloud model")
        self._template_bytes = len(str(info.get("template", "")).encode("utf-8"))
        self.identity = {"backend": "ollama", "model": matches[0]["name"],
                         "digest": matches[0]["digest"], "endpoint": self.endpoint,
                         "local_only": True, "mode": "live", "temperature": 0,
                         "seed": seed, "max_output_tokens": max_output_tokens,
                         "context_tokens": context_tokens, "thinking": False,
                         "structured_decoding": True}

    def _request(self, suffix: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.endpoint + suffix, data=data,
                                         headers={"Content-Type": "application/json"})
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def generate(self, system: str, prompt: str, schema: dict) -> Response:
        # Conservative byte allowance for supported byte-tokenizing local LMs.
        # Reserve the output and chat-template overhead instead of allowing the
        # server to silently truncate a long instruction or private input.
        upper_bound = len(system.encode("utf-8")) + len(prompt.encode("utf-8"))
        upper_bound += 2 * self._template_bytes + 512 + self.max_output_tokens
        if upper_bound > self.context_tokens:
            raise ValueError("Input exceeds conservative context allowance; increase the locked context limit")
        start = time.perf_counter()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False, "think": False,
            "options": {"temperature": 0, "seed": self.seed,
                        "num_predict": self.max_output_tokens, "num_ctx": self.context_tokens},
        }
        if schema:
            body["format"] = schema
        payload = self._request("/api/chat", body)
        usage = {"input_tokens": payload.get("prompt_eval_count"),
                 "output_tokens": payload.get("eval_count")}
        if not payload.get("done") or payload.get("error"):
            raise ModelCallError("Local model did not return a completed response", **usage)
        if payload.get("done_reason") == "length":
            raise ModelCallError("Local output reached its token cap; truncated responses are failures", **usage)
        return Response(text=payload["message"]["content"],
                        input_tokens=payload.get("prompt_eval_count"),
                        output_tokens=payload.get("eval_count"),
                        latency_seconds=time.perf_counter() - start)
