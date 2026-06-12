# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.chat_utils import normalize_xtml_tool_result_messages


def _tool_call(name: str, call_id: str | None = None) -> dict:
    tool_call = {
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }
    if call_id is not None:
        tool_call["id"] = call_id
    return tool_call


def test_normalize_xtml_tool_result_messages_orders_and_fills_attrs():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("lookup", "lookup:0"),
                _tool_call("lookup", "lookup:1"),
            ],
        },
        {"role": "tool", "tool_call_id": "lookup:1", "content": "second"},
        {"role": "tool", "tool_call_id": "lookup:0", "content": "first"},
    ]

    normalized = normalize_xtml_tool_result_messages(messages)

    assert [message["content"] for message in normalized[1:]] == [
        "first",
        "second",
    ]
    assert normalized[1]["tool"] == "lookup"
    assert normalized[1]["index"] == 1
    assert normalized[2]["tool"] == "lookup"
    assert normalized[2]["index"] == 2


def test_normalize_xtml_tool_result_messages_accepts_v4_aliases():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("search"),
                _tool_call("fetch"),
            ],
        },
        {"role": "tool", "tool_call_id": "fetch:1", "content": "b"},
        {"role": "tool", "tool_call_id": "search:0", "content": "a"},
    ]

    normalized = normalize_xtml_tool_result_messages(messages)

    assert [message["content"] for message in normalized[1:]] == ["a", "b"]
    assert normalized[1]["tool"] == "search"
    assert normalized[1]["index"] == 1
    assert normalized[2]["tool"] == "fetch"
    assert normalized[2]["index"] == 2


def test_normalize_xtml_tool_result_messages_rejects_k2_aliases():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("search"),
                _tool_call("fetch"),
            ],
        },
        {"role": "tool", "tool_call_id": "functions.fetch:1", "content": "b"},
    ]

    assert normalize_xtml_tool_result_messages(messages) == messages


def test_normalize_xtml_tool_result_messages_leaves_unknown_block_unchanged():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("lookup", "lookup:0")],
        },
        {"role": "tool", "tool_call_id": "unknown", "content": "result"},
    ]

    assert normalize_xtml_tool_result_messages(messages) == messages
