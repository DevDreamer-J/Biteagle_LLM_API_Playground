"""08_assistant.py 功能测试：流式开关、SDK 切换、闲聊与工具两个功能。

覆盖 2 SDK（openai / dashscope）× 2 流式（开/关）× 2 功能（普通闲聊 / 文本相似度工具）
共 8 种组合，全部走真实 API（模型与密钥配置来自 .env）。

断言策略：
  闲聊 —— 返回文本非空，且全程未发生工具调用（对话中不出现 role=tool 消息）；
  工具 —— 对话中必须出现模型发起的 tool call 与本地执行的 tool result
        （role=tool 消息内容里带 similarity 字段），且最终回答非空。
"""

CHAT_INPUT = "请用一句话介绍你自己"
TOOL_INPUT = "比较这两句话的相似度：我喜欢吃苹果 和 我爱吃苹果"


def _history(assistant, user: str) -> list[dict]:
    """每组测试用独立的对话历史，互不污染。"""
    return [
        {"role": "system", "content": assistant.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _assert_chat_ok(messages: list[dict], text: str) -> None:
    """闲聊断言：回答非空、全程无工具调用。"""
    assert isinstance(text, str) and text.strip(), "最终回答为空"
    assert not [m for m in messages if m.get("role") == "tool"], "闲聊不应触发工具调用"


def _assert_tool_used(messages: list[dict]) -> None:
    """工具断言：出现过 tool call 与本地 tool result，结果里带 similarity 字段。"""
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs, "模型未发起工具调用（对话中没有 role=tool 消息）"
    assert any('"similarity"' in m["content"] for m in tool_msgs), "tool result 缺少 similarity 字段"


# ---------- 普通闲聊：2 SDK × 2 流式 ----------


def test_chat_openai_stream(assistant, client):
    msgs = _history(assistant, CHAT_INPUT)
    text = assistant.run_turn("openai", True, client, msgs)
    _assert_chat_ok(msgs, text)


def test_chat_openai_nostream(assistant, client):
    msgs = _history(assistant, CHAT_INPUT)
    text = assistant.run_turn("openai", False, client, msgs)
    _assert_chat_ok(msgs, text)


def test_chat_dashscope_stream(assistant, client):
    msgs = _history(assistant, CHAT_INPUT)
    text = assistant.run_turn("dashscope", True, None, msgs)
    _assert_chat_ok(msgs, text)


def test_chat_dashscope_nostream(assistant, client):
    msgs = _history(assistant, CHAT_INPUT)
    text = assistant.run_turn("dashscope", False, None, msgs)
    _assert_chat_ok(msgs, text)


# ---------- 文本相似度工具：2 SDK × 2 流式 ----------


def test_tool_openai_stream(assistant, client):
    msgs = _history(assistant, TOOL_INPUT)
    text = assistant.run_turn("openai", True, client, msgs)
    _assert_tool_used(msgs)
    assert text.strip(), "工具调用后的最终回答为空"


def test_tool_openai_nostream(assistant, client):
    msgs = _history(assistant, TOOL_INPUT)
    text = assistant.run_turn("openai", False, client, msgs)
    _assert_tool_used(msgs)
    assert text.strip(), "工具调用后的最终回答为空"


def test_tool_dashscope_stream(assistant, client):
    msgs = _history(assistant, TOOL_INPUT)
    text = assistant.run_turn("dashscope", True, None, msgs)
    _assert_tool_used(msgs)
    assert text.strip(), "工具调用后的最终回答为空"


def test_tool_dashscope_nostream(assistant, client):
    msgs = _history(assistant, TOOL_INPUT)
    text = assistant.run_turn("dashscope", False, None, msgs)
    _assert_tool_used(msgs)
    assert text.strip(), "工具调用后的最终回答为空"
