"""要求 3：结构化输出 —— 用 pydantic 定义目标结构（名字/年龄/个人介绍）并校验模型输出。

失败处理演示：第 1 次调用故意不带 JSON 格式要求，模型返回散文导致校验失败；
随后把错误信息和格式要求回传给模型，重新调用一次，拿到合法 JSON。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
调用记录（输入/输出/Token 用量/耗时）统一由 06_logging 负责，不在本文件处理。
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()  # 从 .env 读取配置

# 重试时回传给模型的格式要求（明确禁止 markdown 代码块）
FORMAT_PROMPT = (
    "请严格只输出一个 JSON 对象的纯文本，不要用 markdown 代码块包裹，"
    "不要输出任何其他文字，包含三个字段："
    "name（名字，字符串）、age（年龄，整数）、introduction（个人介绍，字符串）。"
)


class PersonInfo(BaseModel):
    """结构化输出的目标结构：名字、年龄、个人介绍。"""

    name: str
    age: int
    introduction: str


def get_client() -> OpenAI:
    """创建 OpenAI 客户端，指向阿里云 MaaS 的 OpenAI 兼容接口。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client: OpenAI, model: str, messages: list[dict]) -> str:
    """发起一次普通对话，返回回复文本。"""
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def parse_person(text: str) -> PersonInfo:
    """把模型输出解析并校验成 PersonInfo；格式不对时抛出带原因的 ValueError。"""
    try:
        data = json.loads(text)
        return PersonInfo.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"输出不是合法的目标 JSON：{e}") from e


def structured_chat(client: OpenAI, model: str, messages: list[dict], max_retries: int = 1) -> PersonInfo:
    """带重试的结构化输出：校验失败时把错误信息和格式要求回传模型，重新调用。"""
    current = list(messages)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        text = call_llm(client, model, current)
        try:
            return parse_person(text)
        except ValueError as e:
            last_error = e
            print(f"[第 {attempt + 1} 次输出校验失败] {e}")
            if attempt == max_retries:
                break
            # 关键：把模型的错误输出 + 失败原因 + 格式要求追加进对话，再要一次
            print("[回传错误信息，重新调用模型…]")
            current = current + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"你刚才的输出不符合要求：{e}\n{FORMAT_PROMPT}"},
            ]

    raise RuntimeError(f"重试 {max_retries} 次后仍未获得合法 JSON，最后一次错误：{last_error}")


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
        # 故意不带任何 JSON 格式要求 —— 模型几乎必然返回散文，
        # 从而真实地制造出一次"格式错误"，触发下面的重试链路
        {"role": "user", "content": "请你介绍一下自己。"},
    ]

    print("=== 结构化输出（失败自动重试演示）===")
    try:
        person = structured_chat(client, model, messages)
    except RuntimeError as e:
        print(f"最终失败：{e}")
        return

    print("\n=== pydantic 校验通过 ===")
    print(f"名字：{person.name}")
    print(f"年龄：{person.age}")
    print(f"个人介绍：{person.introduction}")


if __name__ == "__main__":
    main()
