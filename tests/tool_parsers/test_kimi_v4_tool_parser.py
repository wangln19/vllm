# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers.kimi_v4_tool_parser import KimiV4ToolParser


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )


def _arg(key: str, typ: str, value: str) -> str:
    return f'[open]argument key="{key}" type="{typ}"[sep]{value}[close]argument[sep]'


def _call(tool: str, index: int, *args: str) -> str:
    return (
        f'[open]call tool="{tool}" index="{index}"[sep]{"".join(args)}[close]call[sep]'
    )


def _response(content: str) -> str:
    return f"[open]response[sep]{content}[close]response[sep]"


def _tools(*calls: str) -> str:
    return f"[open]tools[sep]{''.join(calls)}[close]tools[sep]"


def test_extract_tool_calls_with_response_and_typed_arguments():
    parser = KimiV4ToolParser(DummyTokenizer())

    output = _response("answer") + _tools(
        _call(
            "calc",
            1,
            _arg("x", "number", "1"),
            _arg("flag", "boolean", "true"),
            _arg("text", "string", "raw"),
        )
    )
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.content == "answer"
    assert len(extracted.tool_calls) == 1
    tool_call = extracted.tool_calls[0]
    assert tool_call.id == "calc:0"
    assert tool_call.function.name == "calc"
    assert json.loads(tool_call.function.arguments) == {
        "x": 1,
        "flag": True,
        "text": "raw",
    }


def test_extract_tool_calls_unescapes_attributes():
    parser = KimiV4ToolParser(DummyTokenizer())

    output = _tools(_call("a&amp;b&quot;c", 1, _arg("k&amp;q", "string", "v")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == 'a&b"c'
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"k&q": "v"}


def test_extract_content_from_whitespace_degraded_markers():
    parser = KimiV4ToolParser(DummyTokenizer())

    extracted = parser.extract_tool_calls(
        "[open] response [sep]answer[close] response [sep]",
        _request(),
    )

    assert extracted.tools_called is False
    assert extracted.content == "answer"


def test_streaming_split_markers_do_not_leak():
    parser = KimiV4ToolParser(DummyTokenizer())
    request = _request()
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    chunks = [
        "[open]",
        "response",
        "[sep]Hi",
        "[open]",
        "tools",
        "[sep]",
        '[open]call tool="calc" index="1"[sep]',
        _arg("x", "number", "1"),
        "[close]call",
        "[sep]",
    ]

    for i, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [i]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[i],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    content = "".join(message.content or "" for message in messages)
    tool_deltas = [
        tool_call for message in messages for tool_call in (message.tool_calls or [])
    ]

    assert content == "Hi"
    assert "[open]" not in content
    assert "[sep]" not in content
    assert len(tool_deltas) == 1
    assert tool_deltas[0].id == "calc:0"
    assert tool_deltas[0].function.name == "calc"
    assert json.loads(tool_deltas[0].function.arguments) == {"x": 1}


def test_adjust_request_keeps_xtml_markers_contiguous():
    parser = KimiV4ToolParser(DummyTokenizer())
    request = _request()

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False
    assert KimiV4ToolParser.supports_required_and_named is False
