"""要求 4：完整工具调用链路 —— 模型发起 tool call → 本地执行函数 → 回传 tool result → 模型生成最终回答。

工具：get_today_date，返回今天日期（如 2026年10月21日）。
对话直接询问"今天日期"——模型自己不知道实时日期，必须调用工具才能回答，
因此它会自主发起 tool call，走完整个链路。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
调用记录（输入/输出/Token 用量/耗时）统一由 06_logging 负责，不在本文件处理。
"""

import datetime
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 读取配置

# 工具声明：告诉模型有哪些工具可用、什么时候该用（模型据此决定是否发起调用）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_today_date",
            "description": "获取今天的日期，返回形如 2026年10月21日 的字符串",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def get_client() -> OpenAI:
    """创建 OpenAI 客户端，指向阿里云 MaaS 的 OpenAI 兼容接口。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_today_date() -> str:
    """本地工具：返回今天的日期，如 2026年10月21日。"""
    today = datetime.date.today()
    return f"{today.year}年{today.month}月{today.day}日"


def execute_tool(name: str) -> str:
    """本地执行工具并返回结果字符串。"""
    if name == "get_today_date":
        return get_today_date()
    return f"未知工具：{name}"


def run_tool_call(client: OpenAI, model: str, messages: list[dict], max_rounds: int = 5) -> str:
    """完整工具调用链路，返回模型的最终回答。"""
    current = list(messages)

    for _ in range(max_rounds):
        resp = client.chat.completions.create(model=model, messages=current, tools=TOOLS)
        msg = resp.choices[0].message

        # 模型不再发起工具调用，说明它已给出最终回答
        if not msg.tool_calls:
            return msg.content

        # 把"模型发起工具调用"的这条 assistant 消息原样加入对话
        current.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        # 逐个执行工具，把结果以 role=tool 回传（tool_call_id 用于对应是哪次调用）
        for tc in msg.tool_calls:
            # 若工具带参数，这里需 json.loads(tc.function.arguments) 后再传给函数
            print(f"[模型发起 tool call] {tc.function.name}({tc.function.arguments})")
            result = execute_tool(tc.function.name)
            print(f"[本地执行结果] {result}")
            current.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    raise RuntimeError(f"超过 {max_rounds} 轮仍在调用工具，已中止")


def main() -> None:
    model = os.getenv("LLM_DEEPSEEK_MODEL")
    if not model:
        raise RuntimeError("请先在 .env 中配置 LLM_DEEPSEEK_MODEL")

    client = get_client()

    messages = [
        {"role": "user", "content": "今天日期是多少？"},
    ]

    print("=== 工具调用链路演示 ===")
    answer = run_tool_call(client, model, messages)
    print("\n=== 模型最终回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
