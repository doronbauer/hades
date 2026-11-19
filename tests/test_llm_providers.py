"""LLM provider configuration and the OpenAI client (SDK mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hades.config import ConfigError, LLMSettings
from hades.llm import INTERPRET_SCHEMA, REFINE_SCHEMA, LLMError, OpenAIClient, make_client


def test_openai_defaults(make_config, music_dir) -> None:
    llm = make_config(music_dir, extra='[llm]\nprovider = "openai"\n').llm
    assert (llm.model, llm.api_key_env, llm.api_key) == ("gpt-6.1-terra", "OPENAI_API_KEY", None)


def test_api_key_from_config_wins_over_environment(make_config, music_dir, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    llm = make_config(music_dir, extra='[llm]\nprovider = "openai"\napi_key = "from-config"\n').llm
    assert llm.resolved_api_key == "from-config"
    assert "from-config" not in repr(llm), "the key must not leak into logs or tracebacks"
    blank = make_config(music_dir, extra='[llm]\nprovider = "openai"\napi_key = " "\n').llm
    assert blank.resolved_api_key == "from-env"


def test_invalid_provider(make_config, music_dir) -> None:
    with pytest.raises(ConfigError, match="llm.provider"):
        make_config(music_dir, extra='[llm]\nprovider = "nope"\n')


def test_missing_openai_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="no OpenAI API key"):
        make_client(LLMSettings(provider="openai", model="gpt-6.1-terra",
                                api_key_env="OPENAI_API_KEY"))


def _strict_problems(schema: dict[str, Any], where: str = "$") -> list[str]:
    """OpenAI strict mode: every object closed, every property required."""
    out = []
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            out.append(f"{where}: additionalProperties must be false")
        if set(schema.get("required", [])) != set(schema.get("properties", {})):
            out.append(f"{where}: all properties must be required")
    for name, sub in schema.get("properties", {}).items():
        out += _strict_problems(sub, f"{where}.{name}")
    for i, sub in enumerate(schema.get("anyOf", [])):
        out += _strict_problems(sub, f"{where}|{i}")
    if "items" in schema:
        out += _strict_problems(schema["items"], f"{where}[]")
    return out


@pytest.mark.parametrize("schema", [INTERPRET_SCHEMA, REFINE_SCHEMA])
def test_schemas_are_strict_mode_compatible(schema: dict) -> None:
    assert _strict_problems(schema) == []


class FakeResponses:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def client_returning(response: Any) -> tuple[OpenAIClient, FakeResponses]:
    client = OpenAIClient(LLMSettings(provider="openai", model="gpt-6.1-terra",
                                      api_key_env="OPENAI_API_KEY", api_key="sk-test"))
    fake = FakeResponses(response)
    client._client = SimpleNamespace(responses=fake)  # type: ignore[assignment]
    return client, fake


def response(text: str | None = '{"a": 1}', status: str = "completed", reason: str | None = None,
             refusal: str | None = None) -> SimpleNamespace:
    content = [SimpleNamespace(type="refusal", refusal=refusal)] if refusal else \
        [SimpleNamespace(type="output_text", text=text)]
    return SimpleNamespace(
        error=None, status=status, output_text=text,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(type="message", content=content)])


def test_openai_request_shape_and_result() -> None:
    client, fake = client_returning(response('{"track_ids": [1], "rationale": "x"}'))
    assert client.complete_json("SYS", "USER", REFINE_SCHEMA) == {"track_ids": [1],
                                                                  "rationale": "x"}
    kw = fake.kwargs
    assert kw["model"] == "gpt-6.1-terra" and kw["instructions"] == "SYS" and kw["input"] == "USER"
    fmt = kw["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["schema"] is REFINE_SCHEMA
    assert kw["reasoning"] == {"effort": "low"} and kw["store"] is False


@pytest.mark.parametrize("resp, message", [
    (response(status="incomplete", reason="max_output_tokens", text=None), "cut off"),
    (response(status="incomplete", reason="content_filter", text=None), "declined"),
    (response(refusal="no", text=None), "declined this request: no"),
    (response(text="not json"), "malformed JSON"),
    (response(text="[1]"), "not an object"),
    (response(text=None), "no answer"),
])
def test_openai_failures_become_llm_errors(resp, message: str) -> None:
    client, _ = client_returning(resp)
    with pytest.raises(LLMError, match=message):
        client.complete_json("s", "u", REFINE_SCHEMA)


def test_openai_connection_error() -> None:
    import httpx2
    import openai

    exc = openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com"))
    client, _ = client_returning(exc)
    with pytest.raises(LLMError, match="could not reach"):
        client.complete_json("s", "u", REFINE_SCHEMA)
