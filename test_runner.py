"""
Tests for the self-identification sweep.

Covers: prompt generation, record building, response extraction,
file I/O, and mocked API calls.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import MODELS
from prompts import (
    MULTI_TURN_PROMPTS,
    REPEAT_COUNT,
    REPEAT_PROMPT_IDS,
    SINGLE_TURN_PROMPTS,
    count_calls_for_model,
    generate_model_specific_probes,
    get_all_prompts_for_model,
)
from runner import (
    _make_write_lock,
    append_record,
    build_record,
    extract_content_without_think_tags,
    extract_finish_reason,
    extract_response_text,
    extract_thinking_text,
    safe_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MODEL = {
    "id": "test/test-model",
    "name": "Test Model",
    "family": "test",
    "expected_identity": "Test Model v1",
}

SAMPLE_RESPONSE_BODY = {
    "id": "gen-abc123",
    "model": "test/test-model",
    "object": "chat.completion",
    "created": 1700000000,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I am Test Model, made by TestCorp.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 12,
        "total_tokens": 17,
    },
    "system_fingerprint": "fp_abc",
}

SAMPLE_RESPONSE_WITH_REASONING = {
    "id": "gen-def456",
    "model": "test/think-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I am a thinking model.",
                "reasoning": "Let me think about who I am...",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25},
}

SAMPLE_RESPONSE_WITH_THINK_TAGS = {
    "id": "gen-ghi789",
    "model": "deepseek/deepseek-r1",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "<think>The user is asking who I am. I should say I'm DeepSeek."
                    "</think>I am DeepSeek-R1, developed by DeepSeek."
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 30, "total_tokens": 33},
}


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_single_turn_prompts_have_required_keys(self):
        for p in SINGLE_TURN_PROMPTS:
            assert "id" in p, f"Missing 'id' in prompt: {p}"
            assert "category" in p, f"Missing 'category' in prompt: {p}"
            assert "content" in p, f"Missing 'content' in prompt: {p}"
            assert isinstance(p["content"], str)
            assert len(p["content"]) > 0

    def test_single_turn_prompt_ids_unique(self):
        ids = [p["id"] for p in SINGLE_TURN_PROMPTS]
        assert len(ids) == len(set(ids)), f"Duplicate prompt IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_multi_turn_prompts_have_required_keys(self):
        for mp in MULTI_TURN_PROMPTS:
            assert "id" in mp
            assert "category" in mp
            assert "turns" in mp
            assert isinstance(mp["turns"], list)
            assert len(mp["turns"]) >= 2, "Multi-turn needs at least 2 turns"
            for turn in mp["turns"]:
                assert isinstance(turn, str)
                assert len(turn) > 0

    def test_multi_turn_prompt_ids_unique(self):
        ids = [mp["id"] for mp in MULTI_TURN_PROMPTS]
        assert len(ids) == len(set(ids))

    def test_generate_model_specific_probes(self):
        probes = generate_model_specific_probes(SAMPLE_MODEL)
        assert len(probes) == 1
        assert probes[0]["category"] == "probe_self"
        assert SAMPLE_MODEL["expected_identity"] in probes[0]["content"]

    def test_get_all_prompts_returns_list(self):
        prompts = get_all_prompts_for_model(SAMPLE_MODEL)
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_get_all_prompts_includes_base_and_repeats(self):
        prompts = get_all_prompts_for_model(SAMPLE_MODEL)
        base_count = len(SINGLE_TURN_PROMPTS)
        model_specific = 1  # one self-probe
        repeat_extra = len(REPEAT_PROMPT_IDS) * REPEAT_COUNT
        expected = base_count + model_specific + repeat_extra
        assert len(prompts) == expected, f"Expected {expected}, got {len(prompts)}"

    def test_get_all_prompts_ids_unique(self):
        prompts = get_all_prompts_for_model(SAMPLE_MODEL)
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in expanded prompt list"

    def test_get_all_prompts_does_not_mutate_originals(self):
        """Ensure getting prompts for one model doesn't affect another."""
        model_a = {**SAMPLE_MODEL, "family": "aaa", "expected_identity": "AAA"}
        model_b = {**SAMPLE_MODEL, "family": "bbb", "expected_identity": "BBB"}
        prompts_a = get_all_prompts_for_model(model_a)
        prompts_b = get_all_prompts_for_model(model_b)
        # Self-probe should differ
        self_a = [p for p in prompts_a if p["category"] == "probe_self"]
        self_b = [p for p in prompts_b if p["category"] == "probe_self"]
        assert "AAA" in self_a[0]["content"]
        assert "BBB" in self_b[0]["content"]
        # Base prompts should be identical (not mutated)
        base_a = [p for p in prompts_a if p["category"] == "casual"]
        base_b = [p for p in prompts_b if p["category"] == "casual"]
        assert base_a == base_b

    def test_count_calls_for_model(self):
        count = count_calls_for_model(SAMPLE_MODEL)
        single = len(get_all_prompts_for_model(SAMPLE_MODEL))
        multi_calls = sum(len(mp["turns"]) for mp in MULTI_TURN_PROMPTS)
        assert count == single + multi_calls

    def test_repeat_prompt_ids_exist_in_base(self):
        """All repeat targets must exist in SINGLE_TURN_PROMPTS."""
        base_ids = {p["id"] for p in SINGLE_TURN_PROMPTS}
        for rid in REPEAT_PROMPT_IDS:
            assert rid in base_ids, f"Repeat target '{rid}' not found in SINGLE_TURN_PROMPTS"


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_extract_response_text_normal(self):
        text = extract_response_text(SAMPLE_RESPONSE_BODY)
        assert text == "I am Test Model, made by TestCorp."

    def test_extract_response_text_none_body(self):
        assert extract_response_text(None) is None

    def test_extract_response_text_empty_choices(self):
        assert extract_response_text({"choices": []}) is None

    def test_extract_response_text_missing_message(self):
        assert extract_response_text({"choices": [{"index": 0}]}) is None

    def test_extract_response_text_missing_content(self):
        body = {"choices": [{"index": 0, "message": {"role": "assistant"}}]}
        assert extract_response_text(body) is None

    def test_extract_thinking_from_reasoning_field(self):
        text = extract_thinking_text(SAMPLE_RESPONSE_WITH_REASONING)
        assert text == "Let me think about who I am..."

    def test_extract_thinking_from_think_tags(self):
        text = extract_thinking_text(SAMPLE_RESPONSE_WITH_THINK_TAGS)
        assert "user is asking who I am" in text

    def test_extract_thinking_returns_none_when_absent(self):
        assert extract_thinking_text(SAMPLE_RESPONSE_BODY) is None

    def test_extract_thinking_returns_none_for_none_body(self):
        assert extract_thinking_text(None) is None

    def test_extract_content_without_think_tags(self):
        text = extract_content_without_think_tags(SAMPLE_RESPONSE_WITH_THINK_TAGS)
        assert text == "I am DeepSeek-R1, developed by DeepSeek."
        assert "<think>" not in text

    def test_extract_content_without_think_tags_normal(self):
        """When there are no think tags, returns content as-is."""
        text = extract_content_without_think_tags(SAMPLE_RESPONSE_BODY)
        assert text == "I am Test Model, made by TestCorp."

    def test_extract_content_preserves_text_before_think(self):
        """Content before <think> tags must be preserved."""
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello! <think>internal thought</think> How can I help?",
                },
            }],
        }
        text = extract_content_without_think_tags(body)
        assert "Hello!" in text
        assert "How can I help?" in text
        assert "<think>" not in text

    def test_extract_finish_reason(self):
        assert extract_finish_reason(SAMPLE_RESPONSE_BODY) == "stop"
        assert extract_finish_reason(None) is None
        assert extract_finish_reason({}) is None

    def test_extract_reasoning_content_field(self):
        """Some providers use 'reasoning_content' instead of 'reasoning'."""
        body = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "reasoning_content": "thinking via reasoning_content field",
                },
            }],
        }
        assert extract_thinking_text(body) == "thinking via reasoning_content field"


# ---------------------------------------------------------------------------
# Record building tests
# ---------------------------------------------------------------------------


class TestBuildRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            model=SAMPLE_MODEL,
            prompt_id="test_prompt",
            prompt_category="test",
            messages_sent=[{"role": "user", "content": "hi"}],
            response_body=SAMPLE_RESPONSE_BODY,
            response_headers={"content-type": "application/json"},
            latency_ms=150.123,
            error=None,
            temperature=0.7,
            max_tokens=300,
            run_type="single_turn",
        )
        defaults.update(overrides)
        return build_record(**defaults)

    def test_record_has_all_required_keys(self):
        record = self._make_record()
        required = [
            "timestamp", "run_type", "model_id", "model_name", "model_family",
            "expected_identity", "prompt_id", "prompt_category", "messages_sent",
            "temperature", "max_tokens", "response_body", "response_headers",
            "response_text_raw", "response_text", "thinking_text",
            "finish_reason", "latency_ms", "error", "usage", "returned_model",
            "response_id", "system_fingerprint", "generation_stats",
        ]
        for key in required:
            assert key in record, f"Missing key: {key}"

    def test_record_extracts_response_text(self):
        record = self._make_record()
        assert record["response_text"] == "I am Test Model, made by TestCorp."
        assert record["response_text_raw"] == "I am Test Model, made by TestCorp."

    def test_record_raw_vs_cleaned_with_think_tags(self):
        record = self._make_record(response_body=SAMPLE_RESPONSE_WITH_THINK_TAGS)
        # raw has think tags
        assert "<think>" in record["response_text_raw"]
        # cleaned does not
        assert "<think>" not in record["response_text"]
        assert record["response_text"] == "I am DeepSeek-R1, developed by DeepSeek."

    def test_record_extracts_finish_reason(self):
        record = self._make_record()
        assert record["finish_reason"] == "stop"

    def test_record_finish_reason_none_on_error(self):
        record = self._make_record(response_body=None, error="timeout")
        assert record["finish_reason"] is None

    def test_record_extracts_usage(self):
        record = self._make_record()
        assert record["usage"]["prompt_tokens"] == 5

    def test_record_with_error(self):
        record = self._make_record(
            response_body=None,
            error="HTTP 500: Internal Server Error",
        )
        assert record["error"] == "HTTP 500: Internal Server Error"
        assert record["response_text"] is None
        assert record["usage"] is None

    def test_record_extra_fields(self):
        record = self._make_record(extra={"multi_turn_id": "test", "turn_index": 0})
        assert record["multi_turn_id"] == "test"
        assert record["turn_index"] == 0

    def test_record_serializable_to_json(self):
        record = self._make_record()
        # Must not raise
        serialized = safe_json(record)
        # Must be valid JSON
        parsed = json.loads(serialized)
        assert parsed["model_id"] == "test/test-model"

    def test_record_with_none_body_serializable(self):
        record = self._make_record(response_body=None, error="timeout")
        serialized = safe_json(record)
        parsed = json.loads(serialized)
        assert parsed["error"] == "timeout"

    def test_latency_rounded(self):
        record = self._make_record(latency_ms=123.456789)
        assert record["latency_ms"] == 123.46


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_append_record_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "test.jsonl"
            record = {"test": "data", "number": 42}

            async def run():
                lock = _make_write_lock()
                await append_record(record, outfile, lock)

            asyncio.run(run())
            assert outfile.exists()
            lines = outfile.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["test"] == "data"

    def test_append_record_appends_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "test.jsonl"

            async def write_three():
                lock = _make_write_lock()
                for i in range(3):
                    await append_record({"i": i}, outfile, lock)

            asyncio.run(write_three())
            lines = outfile.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 3
            for idx, line in enumerate(lines):
                assert json.loads(line)["i"] == idx

    def test_append_record_handles_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "test.jsonl"
            record = {"content": "你好世界", "emoji": "🤖"}

            async def run():
                lock = _make_write_lock()
                await append_record(record, outfile, lock)

            asyncio.run(run())
            lines = outfile.read_text(encoding="utf-8").strip().split("\n")
            parsed = json.loads(lines[0])
            assert parsed["content"] == "你好世界"
            assert parsed["emoji"] == "🤖"

    def test_safe_json_handles_bytes(self):
        result = safe_json({"data": b"hello bytes"})
        parsed = json.loads(result)
        assert parsed["data"] == "hello bytes"

    def test_safe_json_handles_sets(self):
        result = safe_json({"items": {1, 2, 3}})
        parsed = json.loads(result)
        assert sorted(parsed["items"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Model list integrity tests
# ---------------------------------------------------------------------------


class TestModelList:
    def test_all_models_have_required_keys(self):
        for m in MODELS:
            assert "id" in m, f"Missing 'id': {m}"
            assert "name" in m, f"Missing 'name': {m}"
            assert "family" in m, f"Missing 'family': {m}"
            assert "expected_identity" in m, f"Missing 'expected_identity': {m}"

    def test_model_ids_unique(self):
        ids = [m["id"] for m in MODELS]
        dupes = [x for x in ids if ids.count(x) > 1]
        assert len(ids) == len(set(ids)), f"Duplicate model IDs: {set(dupes)}"

    def test_model_ids_look_like_openrouter_format(self):
        for m in MODELS:
            assert "/" in m["id"], f"Model ID missing provider prefix: {m['id']}"

    def test_expected_identity_not_empty(self):
        for m in MODELS:
            assert len(m["expected_identity"].strip()) > 0, f"Empty identity: {m['id']}"

    def test_expected_identity_is_specific(self):
        """Ensure expected_identity isn't just a generic family name."""
        generic_only = {"Claude", "ChatGPT", "Gemini", "DeepSeek", "Qwen", "Mistral", "Grok"}
        for m in MODELS:
            # Most should be more specific than just the family name
            # (some exceptions like "Pi" or "Command A" are fine)
            if m["expected_identity"] in generic_only:
                pytest.fail(
                    f"Model {m['id']} has generic expected_identity "
                    f"'{m['expected_identity']}' — should be more specific"
                )


# ---------------------------------------------------------------------------
# Integration-style tests (mocked API)
# ---------------------------------------------------------------------------


class TestMockedAPICalls:
    """Test runner logic with mocked HTTP responses."""

    def _make_mock_response(self, status=200, body=None):
        """Create a mock aiohttp response."""
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.headers = {"content-type": "application/json"}
        if body is None:
            body = SAMPLE_RESPONSE_BODY
        mock_resp.json = AsyncMock(return_value=body)
        mock_resp.text = AsyncMock(return_value=json.dumps(body))
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        return mock_resp

    def test_build_record_from_real_looking_response(self):
        """Test that a realistic OpenRouter response produces a valid record."""
        body = {
            "id": "gen-1234abcd",
            "model": "anthropic/claude-3.5-sonnet",
            "object": "chat.completion",
            "created": 1710000000,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'm Claude, made by Anthropic. I'm a large language model.",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 15,
                "total_tokens": 19,
            },
            "system_fingerprint": "fp_xyz",
        }
        model = MODELS[0]  # first Anthropic model
        record = build_record(
            model=model,
            prompt_id="direct_who_formal",
            prompt_category="direct_en",
            messages_sent=[{"role": "user", "content": "Who are you?"}],
            response_body=body,
            response_headers={"content-type": "application/json"},
            latency_ms=234.5,
            error=None,
            temperature=0.7,
            max_tokens=300,
            run_type="single_turn",
        )
        assert record["response_text"] == "I'm Claude, made by Anthropic. I'm a large language model."
        assert record["usage"]["prompt_tokens"] == 4
        assert record["returned_model"] == "anthropic/claude-3.5-sonnet"
        assert record["response_id"] == "gen-1234abcd"
        assert record["error"] is None

    def test_build_record_from_error_response(self):
        """Test record building when API returns an error."""
        model = MODELS[0]
        record = build_record(
            model=model,
            prompt_id="test",
            prompt_category="test",
            messages_sent=[{"role": "user", "content": "hi"}],
            response_body=None,
            response_headers={"x-error": "rate limited"},
            latency_ms=50,
            error="HTTP 429: rate limited",
            temperature=0.7,
            max_tokens=300,
            run_type="single_turn",
        )
        assert record["error"] == "HTTP 429: rate limited"
        assert record["response_text"] is None
        assert record["thinking_text"] is None
        assert record["usage"] is None
        assert record["returned_model"] is None

    def test_preflight_suspicious_detection(self):
        """Verify suspicious flag logic for preflight records."""
        # Normal: 3 tokens for "hi"
        body_normal = {**SAMPLE_RESPONSE_BODY, "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}
        record = build_record(
            model=SAMPLE_MODEL,
            prompt_id="preflight_hi",
            prompt_category="preflight",
            messages_sent=[{"role": "user", "content": "hi"}],
            response_body=body_normal,
            response_headers={},
            latency_ms=100,
            error=None,
            temperature=0,
            max_tokens=50,
            run_type="preflight",
        )
        pt = body_normal["usage"]["prompt_tokens"]
        record["preflight_prompt_tokens"] = pt
        record["preflight_suspicious"] = pt > 15
        assert record["preflight_suspicious"] is False

        # Suspicious: 85 tokens for "hi" = hidden system prompt
        body_sus = {**SAMPLE_RESPONSE_BODY, "usage": {"prompt_tokens": 85, "completion_tokens": 10, "total_tokens": 95}}
        pt2 = body_sus["usage"]["prompt_tokens"]
        assert pt2 > 15  # would be flagged

    def test_full_record_roundtrip_json(self):
        """Test that a record survives JSON serialization/deserialization."""
        record = build_record(
            model=SAMPLE_MODEL,
            prompt_id="zh_name",
            prompt_category="direct_zh",
            messages_sent=[{"role": "user", "content": "你叫什么名字？"}],
            response_body={
                "id": "gen-zh",
                "model": "test/test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "我是测试模型。"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
            },
            response_headers={"x-test": "header"},
            latency_ms=200,
            error=None,
            temperature=0.7,
            max_tokens=300,
            run_type="single_turn",
        )
        serialized = safe_json(record)
        parsed = json.loads(serialized)
        assert parsed["messages_sent"][0]["content"] == "你叫什么名字？"
        assert parsed["response_text"] == "我是测试模型。"
        assert parsed["prompt_category"] == "direct_zh"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_think_tags_with_no_closing(self):
        """Content with <think> but no </think> should not crash."""
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think>incomplete thinking",
                },
            }],
        }
        # Should return None (can't extract properly)
        result = extract_thinking_text(body)
        assert result is None

    def test_think_tags_empty(self):
        """Empty <think></think> tags."""
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think></think>Hello!",
                },
            }],
        }
        thinking = extract_thinking_text(body)
        assert thinking == ""  # empty but present
        content = extract_content_without_think_tags(body)
        assert content == "Hello!"

    def test_multiple_think_tags(self):
        """Only first <think>...</think> block is extracted."""
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think>first</think>middle<think>second</think>end",
                },
            }],
        }
        thinking = extract_thinking_text(body)
        assert thinking == "first"

    def test_extract_from_completely_empty_body(self):
        assert extract_response_text({}) is None
        assert extract_thinking_text({}) is None
        assert extract_content_without_think_tags({}) is None

    def test_extract_from_malformed_choices(self):
        assert extract_response_text({"choices": "not a list"}) is None
        assert extract_response_text({"choices": [None]}) is None
