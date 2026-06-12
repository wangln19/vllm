# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_v4_reasoning_parser import KimiV4ReasoningParser


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "[open]think[sep]":
            return [1, 2, 3]
        if text == "[close]think[sep]":
            return [4, 2, 3]
        return [ord(ch) for ch in text]


def test_parser_selection_thinking_disabled():
    parser = KimiV4ReasoningParser(
        DummyTokenizer(), chat_template_kwargs={"enable_thinking": False}
    )

    assert isinstance(parser._identity_parser, IdentityReasoningParser)


def test_extract_reasoning_with_xtml_tags():
    parser = KimiV4ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning(
        "[open]think[sep]step[close]think[sep][open]response[sep]answer",
        request,
    )

    assert reasoning == "step"
    assert content == "[open]response[sep]answer"


def test_extract_reasoning_with_generation_prefix_consumed():
    parser = KimiV4ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning(
        "step[close]think[sep][open]response[sep]answer",
        request,
    )

    assert reasoning == "step"
    assert content == "[open]response[sep]answer"


def test_is_reasoning_end_streaming_uses_full_input_ids():
    parser = KimiV4ReasoningParser(DummyTokenizer())

    assert not parser.is_reasoning_end_streaming([4, 2], [2])
    assert parser.is_reasoning_end_streaming([4, 2, 3], [3])


def test_streaming_split_open_marker_is_held_back():
    parser = KimiV4ReasoningParser(DummyTokenizer())

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="[open]",
        delta_text="[open]",
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
    )
    second = parser.extract_reasoning_streaming(
        previous_text="[open]",
        current_text="[open]think",
        delta_text="think",
        previous_token_ids=[1],
        current_token_ids=[1, 2],
        delta_token_ids=[2],
    )
    third = parser.extract_reasoning_streaming(
        previous_text="[open]think",
        current_text="[open]think[sep]step",
        delta_text="[sep]step",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3, 9],
        delta_token_ids=[3, 9],
    )

    assert first is None
    assert second is None
    assert isinstance(third, DeltaMessage)
    assert third.reasoning == "step"


def test_streaming_split_close_marker_hands_content_downstream():
    parser = KimiV4ReasoningParser(DummyTokenizer())

    previous_text = "[open]think[sep]step"
    partial_close = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=previous_text + "[close]",
        delta_text="[close]",
        previous_token_ids=[1, 2, 3, 9],
        current_token_ids=[1, 2, 3, 9, 4],
        delta_token_ids=[4],
    )
    closed = parser.extract_reasoning_streaming(
        previous_text=previous_text + "[close]",
        current_text=previous_text + "[close]think[sep][open]response[sep]answer",
        delta_text="think[sep][open]response[sep]answer",
        previous_token_ids=[1, 2, 3, 9, 4],
        current_token_ids=[1, 2, 3, 9, 4, 2, 3, 10],
        delta_token_ids=[2, 3, 10],
    )

    assert partial_close is None
    assert isinstance(closed, DeltaMessage)
    assert closed.reasoning is None
    assert closed.content == "[open]response[sep]answer"


def test_adjust_request_keeps_xtml_markers_contiguous():
    parser = KimiV4ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False
