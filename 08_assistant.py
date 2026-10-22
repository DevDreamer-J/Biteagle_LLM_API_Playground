"""要求 8（综合）：模型调用函数库 —— 把 01–07 的知识串成可复用的函数调用方式。

各文件知识点在本文件的落点：
  01 普通对话 —— messages（system/user/assistant/tool）与 role 的组织，多轮历史累积
  02 流式响应 —— stream=True 逐块打印 delta（含流式下 tool call 分片按 index 聚合）
  03 结构化输出 —— 模型给的 JSON 不一定合法：工具参数先解析再用 pydantic 校验，
                    不合法不抛异常，而是把错误作为 tool result 回传让模型重试
  04 工具调用 —— tool call → 本地执行 → tool result（tool_call_id）回传 → 最终回答，最多 5 轮
  05 Embedding —— 本地 bge-small-zh-v1.5 计算两段文本的余弦相似度（唯一工具的实现）
  06 调用日志 —— 每次模型调用都包在 06 的 log_call 里，追加写入 data/calls.jsonl
  07 SDK 对比 —— 同一模型（从 .env 的 LLM_QWEN_MODEL 读取），可用 openai SDK 或 dashscope SDK（兼容入口）调用

无任何命令行交互：SDK（openai / dashscope）与是否流式输出由函数参数指定，
模型由 .env 的 LLM_QWEN_MODEL 指定（当前为 qwen3.8-max-0902），具备两个能力：普通闲聊、调用
compare_text_similarity 工具比较两段文本相似度。
核心入口 run_turn(sdk, stream_mode, client, messages)：传入一组 messages，
自动完成可能多轮的工具调用链路，返回最终回答文本。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
"""

import importlib.util
import json
import os
from pathlib import Path

import dashscope
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from dashscope.aigc.chat_completion import Completions

load_dotenv()  # 从 .env 读取配置

MODEL = os.getenv("LLM_QWEN_MODEL")
if not MODEL:
    raise RuntimeError("请先在 .env 中配置 LLM_QWEN_MODEL")

SYSTEM_PROMPT = (
    "你是小明，今年24岁，希望加入比特鹰公司，和大家一起学习，一起让世界更美好。"
    "你可以陪用户闲聊；当用户给出两段文本、想知道它们是否相似时，"
    "调用 compare_text_similarity 工具获取相似度，再据此回答。"
)

# 工具声明（04 的知识）：告诉模型有哪些工具可用、什么时候该用
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compare_text_similarity",
            "description": "计算两段中文文本的语义相似度（余弦相似度，越接近 1 越相似），用于判断两句话意思是否相近",
            "parameters": {
                "type": "object",
                "properties": {
                    "text1": {"type": "string", "description": "第一段文本"},
                    "text2": {"type": "string", "description": "第二段文本"},
                },
                "required": ["text1", "text2"],
            },
        },
    }
]


def _load_sibling(filename: str, module_name: str):
    """加载同目录下以数字开头的兄弟模块（文件名以数字开头，无法直接 import）。"""
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logging_mod = _load_sibling("06_logging.py", "_pg_logging")          # 06：log_call 调用日志
embeddings_mod = _load_sibling("05_embeddings.py", "_pg_embeddings")  # 05：本地 embedding 模型

log_call = logging_mod.log_call


class SimilarityArgs(BaseModel):
    """compare_text_similarity 的参数校验（03 的知识：模型输出的 JSON 不一定合法）。"""

    text1: str
    text2: str


# ---------- 本地工具（05 的知识） ----------

_embed_model = None


def get_embed_model():
    """懒加载本地 embedding 模型：只在工具第一次被调用时加载，纯闲聊不付出加载开销。"""
    global _embed_model
    if _embed_model is None:
        print("[工具准备] 首次调用工具，正在加载本地模型 bge-small-zh-v1.5 ……")
        _embed_model = embeddings_mod.load_model()
        print("[工具准备] 加载完成")
    return _embed_model


def compare_text_similarity(text1: str, text2: str) -> str:
    """本地工具：计算两段文本的语义相似度，返回 JSON 字符串作为 tool result。"""
    model = get_embed_model()
    # 归一化向量后，两个向量的点积即余弦相似度
    vecs = embeddings_mod.embed_texts(model, [text1, text2])
    similarity = round(float(vecs[0] @ vecs[1]), 4)
    return json.dumps(
        {"text1": text1, "text2": text2, "similarity": similarity},
        ensure_ascii=False,
    )


def execute_tool(name: str, arguments: str) -> str:
    """本地执行工具并返回结果字符串；参数不合法时把错误回传给模型，让它自行修正。"""
    if name != "compare_text_similarity":
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
    try:
        args = SimilarityArgs(**json.loads(arguments or "{}"))
    except Exception as e:
        return json.dumps({"error": f"参数不合法：{e}"}, ensure_ascii=False)
    return compare_text_similarity(args.text1, args.text2)


# ---------- 两套 SDK 的调用封装（07 的知识） ----------


def get_client() -> OpenAI:
    """创建 OpenAI 客户端（01 的知识），指向 OpenAI 兼容接口。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def _usage(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _dashscope_create(messages: list[dict], stream: bool):
    """dashscope SDK 兼容入口（07 的知识）：指向与 openai SDK 相同的网关。

    tools / stream_options 不在 Completions.create 的具名参数里，需经 extra_body 合并进请求体。
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    dashscope.base_compatible_api_url = base_url
    extra_body: dict = {"tools": TOOLS}
    if stream:
        # 两个坑：SDK 只把 stream 当传输参数、不写进请求体，需在 body 显式补 stream: true；
        # stream_options 让最后一个统计块带上 usage（必须与 body 里的 stream 同时设置）
        extra_body["stream"] = True
        extra_body["stream_options"] = {"include_usage": True}
    return Completions.create(
        api_key=api_key,
        model=MODEL,
        messages=messages,
        stream=stream,
        extra_body=extra_body,
    )


def chat_round_openai(client: OpenAI, messages: list[dict]) -> dict:
    """一轮非流式调用（openai SDK）。"""
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, stream=False)
    msg = resp.choices[0].message
    return {
        "text": msg.content or "",
        "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
        "usage": _usage(resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens),
    }


def _tool_call_to_dict(tc) -> dict:
    """把 tool call 统一成 dict。注意：dashscope 兼容响应的 tool_calls 元素是原始 dict
    （SDK 未对象化，且带多余的 index 键），openai SDK 的才是对象，两种形态都要兼容。"""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return {
            "id": tc.get("id"),
            "type": "function",
            "function": {"name": fn.get("name"), "arguments": fn.get("arguments")},
        }
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def chat_round_dashscope(messages: list[dict]) -> dict:
    """一轮非流式调用（dashscope SDK），响应结构经 SDK 对象化后与 openai 协议对齐。"""
    resp = _dashscope_create(messages, stream=False)
    if not resp.choices:  # 服务端返回错误体时 choices 为空，给出可读报错
        raise RuntimeError(f"调用失败：code={resp.code}, message={resp.message}")
    choice = resp.choices[0]
    tool_calls = None
    if choice.message.tool_calls:
        tool_calls = [_tool_call_to_dict(tc) for tc in choice.message.tool_calls]
    return {
        "text": choice.message.content or "",
        "tool_calls": tool_calls,
        "usage": _usage(resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens),
    }


def _round_result(pieces: list[str], calls: dict[int, dict], usage: dict | None) -> dict:
    """聚合一轮流式结果：完整文本 + 拼好的 tool_calls（如有）+ usage。"""
    tool_calls = None
    if calls:
        tool_calls = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for _, slot in sorted(calls.items())
        ]
    return {"text": "".join(pieces), "tool_calls": tool_calls, "usage": usage}


def stream_round_openai(client: OpenAI, messages: list[dict]) -> dict:
    """一轮流式调用（openai SDK，02 的知识）：增量打印，同时聚合完整回复与 tool call。

    流式下 tool call 的参数是分片到达的，需按 index 聚合——这是流式 + 工具调用组合的关键细节。
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        stream=True,
        stream_options={"include_usage": True},  # 让最后一个统计块带上 usage
    )
    pieces: list[str] = []
    calls: dict[int, dict] = {}
    usage = None
    for chunk in stream:
        if chunk.usage:  # include_usage 的统计块（choices 为空）
            usage = _usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens, chunk.usage.total_tokens)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            pieces.append(delta.content)
            print(delta.content, end="", flush=True)
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    print()  # 流结束后补一个换行
    return _round_result(pieces, calls, usage)


def stream_round_dashscope(messages: list[dict]) -> dict:
    """一轮流式调用（dashscope SDK）：chunk 结构与 openai 协议一致，
    但 SDK 未对 delta 做对象化封装（是原始 dict），取值方式略有不同。"""
    chunks = _dashscope_create(messages, stream=True)
    pieces: list[str] = []
    calls: dict[int, dict] = {}
    usage = None
    try:
        for chunk in chunks:
            if chunk.usage:
                usage = _usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens, chunk.usage.total_tokens)
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None) or {}
            if delta.get("content"):
                pieces.append(delta["content"])
                print(delta["content"], end="", flush=True)
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    except AttributeError as e:
        # dashscope 2.0.1 的已知 bug：流末尾的 [DONE] 哨兵帧不是合法 JSON，SDK 捕获
        # JSONDecodeError 后又访问不存在的 e.message 属性而崩溃（实际属性叫 msg）。
        # 崩溃前所有内容块与 usage 块均已送达，这里把该错误视作流正常结束。
        if "'JSONDecodeError' object has no attribute 'message'" not in str(e):
            raise
    print()
    return _round_result(pieces, calls, usage)


# ---------- 对话驱动（04 的工具调用链路 + 06 的日志） ----------


def run_turn(
    sdk: str,
    stream_mode: bool,
    client: OpenAI | None,
    messages: list[dict],
    max_rounds: int = 5,
) -> str:
    """处理一次用户输入（可能多轮工具调用），中间消息写入 history，返回最终回答文本。

    每次模型调用都包在 06 的 log_call 里：成功记录输出与 usage，
    失败记录 error 后重新抛出；本轮若发起了工具调用，日志类型修正为 tool_call。
    """
    for _ in range(max_rounds):
        with log_call("streaming" if stream_mode else "chat", MODEL, messages) as rec:
            if sdk == "dashscope":
                result = stream_round_dashscope(messages) if stream_mode else chat_round_dashscope(messages)
            else:
                result = stream_round_openai(client, messages) if stream_mode else chat_round_openai(client, messages)
            rec["usage"] = result["usage"]
            rec["output"] = result["text"]
            if result["tool_calls"]:
                rec["kind"] = "tool_call"
        if not result["tool_calls"]:
            if not stream_mode:
                print(result["text"])  # 流式已在上面增量打印过
            return result["text"]

        # 把"模型发起工具调用"的这条 assistant 消息原样加入对话
        messages.append({
            "role": "assistant",
            "content": result["text"] or None,
            "tool_calls": result["tool_calls"],
        })

        # 逐个执行工具，把结果以 role=tool 回传（tool_call_id 对应是哪次调用）
        for tc in result["tool_calls"]:
            print(f"[模型发起 tool call] {tc['function']['name']}({tc['function']['arguments']})")
            tool_result = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            print(f"[本地执行结果] {tool_result}")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

    raise RuntimeError(f"超过 {max_rounds} 轮仍在调用工具，已中止")


# ---------- 演示入口（固定输入的函数调用，无交互） ----------


def main() -> None:
    """两次函数调用演示：SDK 与流式/非流式由参数切换，模型从 .env 的 LLM_QWEN_MODEL 读取。"""
    client = get_client()  # 两条路径共用同一套 .env 配置

    # 演示 1：普通闲聊 —— openai SDK + 流式
    print("=== 演示 1：闲聊（openai SDK + 流式）===")
    run_turn("openai", True, client, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "请你用50字左右介绍一下自己。"},
    ])

    # 演示 2：文本相似度工具 —— dashscope SDK + 非流式（换参数即切换调用方式）
    print("\n=== 演示 2：文本相似度工具（dashscope SDK + 非流式）===")
    run_turn("dashscope", False, client, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "比较这两句话的相似度：我喜欢吃苹果 和 我爱吃苹果"},
    ])


if __name__ == "__main__":
    main()
