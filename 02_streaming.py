"""要求 2：流式响应 —— stream=True 逐块接收回复并增量打印（OpenAI 协议的 delta）。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
两套协议流式事件格式的差异（choices[].delta.content / output 增量）在 07_provider_compare 中对比。
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


def stream_chat(client: OpenAI, model: str, messages: list[dict]) -> str:
    """流式发起对话：逐块接收增量内容，实时打印并返回完整回复文本。"""
    # stream=True 时返回的是一个迭代器：每个 chunk 携带一小段增量（delta），
    # 而不是等整个回复生成完再一次性返回
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)

    pieces: list[str] = []
    for chunk in stream:
        # 部分 chunk 不带 choices（如结尾的统计块），需要判空跳过
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            pieces.append(delta.content)
            print(delta.content, end="", flush=True)

    print()  # 流结束后补一个换行
    return "".join(pieces)


def main() -> None:
    model = os.getenv("LLM_DEEPSEEK_MODEL")
    if not model:
        raise RuntimeError("请先在 .env 中配置 LLM_DEEPSEEK_MODEL")

    client = get_client()

    messages = [
        {
            "role": "system",
            "content": "你是小明，今年24岁，希望加入比特鹰公司，和大家一起学习，一起让世界更美好。",
        },
        # 要求更长回复，流式输出的逐块效果更明显
        {"role": "user", "content": "请你用300字左右介绍一下自己。"},
    ]

    print("=== 流式回复（逐块输出）===")
    text = stream_chat(client, model, messages)
    print(f"\n=== 流结束：共收到 {len(text)} 个字符 ===")


if __name__ == "__main__":
    main()
