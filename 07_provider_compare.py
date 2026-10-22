"""要求 7：调用方式对比 —— 同一模型（.env 中的 LLM_QWEN_MODEL，当前为 qwen3.8-max-0902）分别通过 openai SDK 与 DashScope SDK 调用，均为非流式。

对比维度：响应标识、usage 口径、单次请求耗时、回复文本。两条路径共用同一组 messages 与同一网关，保证对比公平。

重要实测结论：dashscope SDK 的原生入口 Generation.call（/api/v1/services/aigc/text-generation/generation）
只路由百炼标准公共 qwen 模型（qwen-plus / qwen-flash 能通过 URL 校验、进入配额校验）；
DeepSeek 等第三方模型与部署快照模型（qwen3.8-max-0902、deepseek-v4.1-flash-1017）都没有原生路由
（报 url error），因此本模型两条路径都只能走 OpenAI 兼容接口——openai SDK 用 chat.completions.create，
dashscope SDK 用 Completions.create 兼容入口，差异只在 SDK 封装层。
"""

import os
import time

import dashscope
from dotenv import load_dotenv
from openai import OpenAI
from dashscope.aigc.chat_completion import Completions

load_dotenv()

MODEL = os.getenv("LLM_QWEN_MODEL")
if not MODEL:
    raise RuntimeError("请先在 .env 中配置 LLM_QWEN_MODEL")


def get_openai_client() -> OpenAI:
    """OpenAI 兼容协议客户端。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_via_openai(client: OpenAI, messages: list[dict]) -> dict:
    """路径 A：OpenAI 兼容协议，非流式。"""
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
    )
    elapsed = round(time.perf_counter() - start, 3)

    choice = resp.choices[0]
    return {
        "text": choice.message.content or "",
        "request_id": resp.id,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
        "elapsed": elapsed,
    }


def call_via_dashscope(messages: list[dict]) -> dict:
    """路径 B：DashScope SDK（Completions.create 兼容模式入口），非流式。

    实测：dashscope 原生入口 Generation.call 只路由百炼标准公共 qwen 模型（qwen-plus /
    qwen-flash 可达、报配额 403），qwen3.8-max-0902 等部署快照模型没有原生路由
    （报 url error），因此本模型经 DashScope SDK 只能走兼容模式入口。
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")

    # SDK 兼容入口默认指向官方地址；这里指向与路径 A 相同的网关，保证对比变量只有 SDK 本身
    dashscope.base_compatible_api_url = base_url

    start = time.perf_counter()
    resp = Completions.create(api_key=api_key, model=MODEL, messages=messages, stream=False)
    elapsed = round(time.perf_counter() - start, 3)

    choice = resp.choices[0]
    return {
        "text": choice.message.content or "",
        "request_id": resp.id,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
        "elapsed": elapsed,
    }


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": "你是小明，今年24岁，希望加入比特鹰公司，和大家一起学习，一起让世界更美好。",
        },
        {"role": "user", "content": "请你用50字左右介绍一下自己。"},
    ]

    client = get_openai_client()

    print("=== 路径 A：OpenAI 兼容协议（非流式） ===")
    r_openai = None
    try:
        r_openai = call_via_openai(client, messages)
        print(f"request_id   = {r_openai['request_id']}")
        print(f"finish_reason= {r_openai['finish_reason']}")
        print(f"usage        = {r_openai['usage']}")
        print(f"elapsed      = {r_openai['elapsed']}s")
        print(f"text         = {r_openai['text']}")
    except Exception as e:
        print(f"调用失败：{type(e).__name__}: {e}")

    print("\n=== 路径 B：DashScope SDK（兼容模式入口，非流式） ===")
    r_dashscope = None
    try:
        r_dashscope = call_via_dashscope(messages)
        print(f"request_id   = {r_dashscope['request_id']}")
        print(f"finish_reason= {r_dashscope['finish_reason']}")
        print(f"usage        = {r_dashscope['usage']}")
        print(f"elapsed      = {r_dashscope['elapsed']}s")
        print(f"text         = {r_dashscope['text']}")
    except Exception as e:
        print(f"调用失败：{type(e).__name__}: {e}")

    if not (r_openai and r_dashscope):
        print("\n有路径调用失败，无法完成并排对比")
        return

    print("\n=== 两种调用方式对比结论 ===")
    print(
        "- SDK 层：openai SDK 用 client.chat.completions.create；dashscope SDK 用 Completions.create；"
        "两者走同一 OpenAI 兼容接口、同一模型，差异仅在 SDK 封装"
    )
    print(
        "- 原生协议差异（实测踩坑）：DashScope 原生入口 Generation.call 走 /services/aigc/text-generation/generation，"
        "响应为 output.choices、usage.input/output_tokens、request_id；但原生路由只覆盖百炼标准公共 qwen 模型"
        "（qwen-plus/qwen-flash 可达），DeepSeek 与 qwen3.8-max-0902 等部署模型均无原生路由（url error），"
        "故两条路径都只能走兼容协议"
    )
    print(
        f"- 响应标识：openai → id={r_openai['request_id']}；dashscope → id={r_dashscope['request_id']}"
    )
    print(f"- 结束原因：{r_openai['finish_reason']} vs {r_dashscope['finish_reason']}")
    print(
        f"- 用量口径：同一兼容协议字段一致，prompt_tokens={r_openai['usage']['prompt_tokens']} vs "
        f"{r_dashscope['usage']['prompt_tokens']}，completion_tokens={r_openai['usage']['completion_tokens']} vs "
        f"{r_dashscope['usage']['completion_tokens']}（差异来自两次独立采样的生成长度）"
    )
    print(f"- 耗时：{r_openai['elapsed']}s vs {r_dashscope['elapsed']}s")
    print(
        f"- 回复长度：{len(r_openai['text'])} 字符 vs {len(r_dashscope['text'])} 字符"
    )
    print(
        "- 文本一致性：两次调用为独立采样，文本通常不完全相同，但语义应一致；若需严格对比生成质量，可将 temperature 设为 0"
    )


if __name__ == "__main__":
    main()
