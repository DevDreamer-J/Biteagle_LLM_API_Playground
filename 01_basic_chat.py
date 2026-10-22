"""要求 1：普通对话调用 —— 构造 messages（system/user 两种 role），打印模型回复。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
调用记录（输入/输出/Token 用量/耗时）统一由 06_logging 负责，不在本文件处理。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 读取配置


def get_client() -> OpenAI:
    """创建 OpenAI 客户端，指向阿里云 MaaS 的 OpenAI 兼容接口。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def basic_chat(client: OpenAI, model: str, messages: list[dict]) -> str:
    """发起一次普通对话，返回模型回复文本。"""
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def main() -> None:
    model = os.getenv("LLM_DEEPSEEK_MODEL")
    if not model:
        raise RuntimeError("请先在 .env 中配置 LLM_DEEPSEEK_MODEL")

    client = get_client()

    # message 是对话的最小单位，role 标明这条消息来自谁：
    #   system —— 设定模型的人设与行为约束，优先级高于 user
    #   user   —— 用户的输入
    messages = [
        {
            "role": "system",
            "content": "你是小明，今年24岁，希望加入比特鹰公司，和大家一起学习，一起让世界更美好。",
        },
        {"role": "user", "content": "请你介绍一下自己。"},
    ]

    print("=== 模型回复 ===")
    print(basic_chat(client, model, messages))


if __name__ == "__main__":
    main()
