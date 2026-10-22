"""要求 6：调用日志 —— 记录每次调用的模型、输入、输出、Token 用量、耗时，追加写入 data/calls.jsonl。

本文件同时是横切关注点的单点实现：导出 log_call 上下文管理器供 01–05 导入，
把 API 调用包进 with 块即可自动记录，避免日志代码重复 5 份。
main() 用一次真实对话演示完整记录流程：成功调用与失败调用各写一条记录。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 读取配置

LOG_PATH = Path(__file__).parent / "data" / "calls.jsonl"


@contextmanager
def log_call(kind: str, model: str, messages: list[dict] | None = None):
    """记录一次 API 调用的上下文管理器（供 01–05 导入使用）。

    用法：
        with log_call("chat", model, messages) as rec:
            resp = client.chat.completions.create(model=model, messages=messages)
            rec["output"] = resp.choices[0].message.content
            rec["usage"] = {"prompt_tokens": .., "completion_tokens": .., "total_tokens": ..}

    离开 with 块时自动计算耗时并追加写入 data/calls.jsonl；
    调用中途抛异常也会把错误写入 error 字段后再重新抛出。
    """
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),  # 记录时刻
        "kind": kind,      # 调用类型：chat / streaming / structured / tool_call / embedding
        "model": model,    # 模型名
        "input": messages, # 输入：完整 messages 列表
        "output": None,    # 输出：模型回复文本
        "usage": None,     # Token 用量：prompt / completion / total
        "error": None,     # 失败时的错误信息
        "elapsed": None,   # 耗时（秒）
    }
    start = time.perf_counter()
    try:
        yield record
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["elapsed"] = round(time.perf_counter() - start, 3)
        append_record(record)


def append_record(record: dict) -> None:
    """把一条记录追加写入 data/calls.jsonl（一行一个 JSON 对象）。"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_calls() -> list[dict]:
    """读取全部历史记录（供汇总统计与 tests 使用）。"""
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_client() -> OpenAI:
    """创建 OpenAI 客户端，指向阿里云 MaaS 的 OpenAI 兼容接口。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


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
        {"role": "user", "content": "请你介绍一下自己。"},
    ]

    # 演示 1：成功调用 —— 完整记录模型、输入、输出、Token 用量、耗时
    print("=== 演示 1：成功调用并记录 ===")
    with log_call("chat", model, messages) as rec:
        resp = client.chat.completions.create(model=model, messages=messages)
        rec["output"] = resp.choices[0].message.content
        rec["usage"] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    print(rec["output"])
    print(f"\n已记录：{rec['usage']}，耗时 {rec['elapsed']}s")

    # 演示 2：失败调用 —— 模型名写错，error 字段同样会被记录
    print("\n=== 演示 2：失败调用也会被记录（模型名故意写错）===")
    try:
        with log_call("chat", "wrong-model-name", messages) as rec:
            resp = client.chat.completions.create(model="wrong-model-name", messages=messages)
            rec["output"] = resp.choices[0].message.content
    except Exception as e:
        print(f"调用失败（已记录 error）：{type(e).__name__}")

    # 演示 3：读回历史记录
    print("\n=== data/calls.jsonl 当前内容 ===")
    for row in read_calls():
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
