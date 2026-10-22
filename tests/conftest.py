"""pytest 共享夹具：加载 08_assistant.py（数字开头的文件名无法直接 import）。"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent  # llm-api-playground 目录


@pytest.fixture(scope="session")
def assistant():
    """以模块方式加载 08_assistant.py，整个测试会话只加载一次。"""
    spec = importlib.util.spec_from_file_location("_pg_assistant_08", ROOT / "08_assistant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def client(assistant):
    """OpenAI 客户端。dashscope 路径不使用它，调用时传 None 即可。"""
    return assistant.get_client()
