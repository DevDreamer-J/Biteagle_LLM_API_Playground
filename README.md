# LLM API Playground

从 0 到 1 实现的大模型 API 调用 Playground，覆盖 LLM API 的核心能力，用于理解 message、role、tool call、stream、usage 等核心概念。

## 项目结构

```
llm-api-playground/
├── 01_basic_chat.py             # 要求1：普通对话（system/user 双 role）
├── 02_streaming.py              # 要求2：流式响应（delta 增量打印）
├── 03_structured.py             # 要求3：结构化输出（pydantic 校验 + 失败重试）
├── 04_tool_call.py              # 要求4：工具调用完整链路（get_today_date 工具）
├── 05_embeddings.py             # 要求5：本地 Embedding 生成与余弦相似度
├── 06_logging.py                # 要求6：调用日志（log_call 上下文管理器）
├── 07_provider_compare.py       # 要求7：调用方式对比（openai SDK vs dashscope SDK）
├── 08_assistant.py              # 要求8（综合）：函数式调用助手，串联 01–07 全部知识点
├── tests/                       # pytest 功能测试（覆盖 08 的 8 种调用组合）
│   ├── conftest.py              # 夹具：加载 08 模块、共享 OpenAI 客户端
│   └── test_08_assistant.py     # 2 SDK × 2 流式 × 闲聊/工具
├── requirements.txt             # 依赖清单（版本锁定）
├── .env / .env.example          # 配置文件（.env 已被 .gitignore 忽略，不入库）
├── data/calls.jsonl             # 调用日志（运行 06 后生成）
└── embedding/bge-small-zh-v1.5/ # 本地 Embedding 模型（BAAI/bge-small-zh-v1.5，不入库，按环境准备下载）
```

## 环境准备

1. Python 3.14（conda 环境 `python14`）
2. 安装依赖：

   ```bash
   python -m pip install -r requirements.txt
   ```

3. 复制 `.env.example` 为 `.env`，填入真实配置。双模型分工：**01–06 读取 `LLM_DEEPSEEK_MODEL`，07/08 读取 `LLM_QWEN_MODEL`**：

   ```ini
   # 01–06 基础示例使用的模型
   LLM_DEEPSEEK_MODEL=deepseek-v4.1-flash-1017
   # 07/08（调用方式对比、综合助手）使用的模型
   LLM_QWEN_MODEL=qwen3.8-max-0902
   API_KEY=sk-xxx
   OpenAI_BASE_URL=https://xxx/compatible-mode/v1
   DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
   ```

4. 下载本地 Embedding 模型（05、08 与 tests 运行时需要；模型二进制不入库，需自行下载约 180MB）：

   ```bash
   python -m pip install modelscope
   python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='embedding/bge-small-zh-v1.5')"
   ```

   脚本以相对路径 `embedding/bge-small-zh-v1.5/` 离线加载，运行时无需联网、无需 API Key。

## 运行说明

| 命令 | 说明 |
|---|---|
| `python 01_basic_chat.py` | 普通对话：构造 messages（system + user），打印模型回复 |
| `python 02_streaming.py` | 流式响应：`stream=True` 逐块接收 delta 并增量打印 |
| `python 03_structured.py` | 结构化输出：pydantic 校验模型返回的 JSON，不合格自动重试 |
| `python 04_tool_call.py` | 工具调用链路：模型发起 tool call → 本地执行 get_today_date → 回传 tool result → 生成最终回答 |
| `python 05_embeddings.py` | 本地加载 bge-small-zh-v1.5 生成向量并计算相似度矩阵（纯离线，无需网络） |
| `python 06_logging.py` | 调用日志演示：成功/失败调用均写入 data/calls.jsonl |
| `python 07_provider_compare.py` | 同一模型分别用 openai SDK 与 dashscope SDK 调用，并排对比 |
| `python 08_assistant.py` | 综合练习：函数式调用（无交互），SDK（openai/dashscope）与流式/非流式由参数切换；闲聊 + 文本相似度工具（本地 bge 模型），调用自动记日志 |

## 核心概念

- **message**：对话的最小单位，一次请求就是一组 message 列表，模型基于全部历史生成回复。
- **role**：消息来源。`system` 设定模型人设与行为约束（优先级高于 user）；`user` 是用户输入；`assistant` 是模型回复；`tool` 是工具执行结果。
- **tool call**：模型判断需要外部信息时，不直接编造答案，而是返回"请帮我执行某函数及参数"的请求，由调用方决定是否执行。
- **tool result**：本地执行工具后，把结果以 `tool` role 消息（携带对应 `tool_call_id`）回传给模型，模型据此生成最终回答。
- **stream**：`stream=True` 时服务端通过 SSE 持续推送增量（delta），客户端逐块拼接、实时展示，首字延迟低；非流式则等全部生成完一次性返回。
- **usage**：Token 用量统计，`prompt_tokens`（输入）+ `completion_tokens`（输出）= `total_tokens`，是计费与上下文窗口控制的依据。

## 示例请求（OpenAI 兼容接口）

```bash
curl -X POST "{OpenAI_BASE_URL}/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-max-0902",
    "messages": [
      {"role": "system", "content": "你是小明，今年24岁。"},
      {"role": "user", "content": "请你介绍一下自己。"}
    ]
  }'
```

响应（节选）：

```json
{
  "id": "chatcmpl-xxxx",
  "model": "qwen3.8-max-0902",
  "choices": [
    {
      "finish_reason": "stop",
      "message": {"role": "assistant", "content": "大家好，我是小明……"}
    }
  ],
  "usage": {"prompt_tokens": 83, "completion_tokens": 348, "total_tokens": 431}
}
```

## 两套调用方式对比结论（07 实测）

1. **SDK 层**：openai SDK 用 `client.chat.completions.create`；dashscope SDK 用 `Completions.create`（兼容模式入口）。两者走同一 OpenAI 兼容接口、同一模型，鉴权方式一致（`Authorization: Bearer`），差异只在 SDK 封装。
2. **协议层**：DashScope 原生入口 `Generation.call`（`POST /api/v1/services/aigc/text-generation/generation`）请求按 `input.messages` / `parameters` 分层，响应为 `output.choices[].message`、`usage.input_tokens / output_tokens`（无 total）、`request_id`——与 OpenAI 协议的顶层 `messages`、`choices[].message`、`prompt/completion/total_tokens`、`id` 一一对应。
3. **路由范围**：DashScope 原生入口只路由百炼标准公共 qwen 模型（qwen-plus / qwen-flash 原生可达）；DeepSeek 等第三方模型与部署快照模型（qwen3.8-max-0902）均无原生路由。详见下方踩坑经历。
4. **公平性验证**：同一组 messages 经两条路径调用，`prompt_tokens` 完全一致（同一 tokenizer）；`completion_tokens` 差异来自两次独立采样。

## 踩坑经历：DeepSeek 模型无法使用 dashscope 原生接口

### 现象

07 最初计划用 dashscope SDK 的原生入口 `Generation.call` 调用 `deepseek-v4.1-flash-1017`，无论把地址指向专属网关还是官方 `dashscope.aliyuncs.com`，都返回 400：

```
{"request_id": "...", "code": "InvalidParameter", "message": "url error, please check url！"}
```

### 排查

1. **鉴权没问题**：同一把 key 访问官方 `/compatible-mode/v1/models` 返回 200，模型列表正常。
2. **URL 没写错**：请求路径与 SDK 源码一致（`/api/v1` + `/services/aigc/text-generation/generation`），且换 URL 变体无济于事。
3. **模型对照实验（关键）**：同一个原生入口，换模型名结果完全不同——

   | 模型 | 原生入口结果 | 说明 |
   |---|---|---|
   | qwen-plus / qwen-flash | 通过 URL 校验，报配额 403 | 原生路由**可达** |
   | qwen3.8-max-0902 / qwen3.8-max | `url error` | 无原生路由 |
   | deepseek-v4.1-flash-1017 | `url error` | 无原生路由 |

### 结论

- **百炼云的 DeepSeek 模型（第三方部署模型）无法使用 dashscope 原生接口，只能使用 OpenAI 兼容接口**。
- **dashscope 原生入口只服务百炼标准公共 qwen 模型**（qwen-plus、qwen-flash 这类模型名），qwen 家族里只有它们两种方式都支持。
- **qwen3.8-max-0902 属于部署快照模型，实测同样没有原生路由**，只能走 OpenAI 兼容接口——快照模型与 DeepSeek 模型在这一点上是同一类。
- 最终方案：07 路径 B 改用 dashscope SDK 的兼容模式入口 `Completions.create`，与路径 A 同网关、同模型对比，两套 SDK 都能真实跑通。

## 调用日志

`data/calls.jsonl` 每行一条 JSON 记录，字段包括：

| 字段 | 说明 |
|---|---|
| time | 记录时刻 |
| kind | 调用类型：chat / streaming / structured / tool_call / embedding |
| model | 模型名 |
| input / output | 完整 messages 输入与模型回复 |
| usage | Token 用量 |
| error | 失败时的错误信息（成功为 null） |
| elapsed | 耗时（秒） |

失败调用写入 `error` 字段后异常会重新抛出，不会静默吞掉。

## 测试用例

tests/：pytest 功能测试，覆盖 08 的 2 SDK（openai/dashscope）× 2 流式（开/关）× 2 功能（闲聊/文本相似度工具）共 8 种组合，运行 `python -m pytest tests -v`。
