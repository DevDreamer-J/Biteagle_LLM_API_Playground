"""要求 5：Embedding 生成与相似度计算 —— 本地加载开源模型 bge-small-zh-v1.5，
把一组文本向量化，计算两两余弦相似度并找出语义最接近的文本对。

与 01-04 不同：本文件不调用云端 API、不需要 .env，模型从本地
embedding/bge-small-zh-v1.5/ 目录加载，完全离线运行。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行，便于 tests 导入。
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# 本地模型目录（相对于本文件，从任何工作目录运行都能找到）
MODEL_DIR = Path(__file__).parent / "embedding" / "bge-small-zh-v1.5"


def load_model() -> SentenceTransformer:
    """加载本地 bge-small-zh-v1.5 模型。"""
    if not MODEL_DIR.exists():
        raise RuntimeError(f"未找到本地模型目录：{MODEL_DIR}")
    return SentenceTransformer(str(MODEL_DIR))


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """把一组文本向量化，返回 (n, dim) 矩阵。

    normalize_embeddings=True：把向量归一化为单位长度，
    之后两个向量的点积就直接等于余弦相似度。
    """
    return model.encode(texts, normalize_embeddings=True)


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵。已归一化的向量，两两点积即余弦相似度。"""
    return vecs @ vecs.T


def main() -> None:
    model = load_model()

    # 一组演示文本：0、1、4 语义相近（都关于密码），2、3 无关
    texts = [
        "如何重置我的登录密码？",
        "忘记密码了怎么办？",
        "今天中午吃什么好？",
        "比特鹰是一家怎样的公司？",
        "找回账号密码的步骤",
    ]

    vecs = embed_texts(model, texts)
    print(f"模型：bge-small-zh-v1.5（本地离线运行）")
    print(f"文本数：{len(texts)}，向量维度：{vecs.shape[1]}\n")

    print("每条文本的向量（前 5 维预览）：")
    for text, vec in zip(texts, vecs):
        preview = ", ".join(f"{v:+.4f}" for v in vec[:5])
        print(f"  {text}  ->  [{preview}, ...]")

    sim = cosine_similarity_matrix(vecs)
    print("\n两两余弦相似度矩阵（对角线恒为 1.0）：")
    header = "        " + " ".join(f"[{j}]{v:>6.3f}" for j, v in enumerate(sim[0]))
    print(header)
    for i, row in enumerate(sim):
        cells = " ".join(f"{v:>7.3f}" for v in row)
        print(f"[{i}]{texts[i][:8]:　<10}{cells}")

    print("\n每条文本语义最接近的其它文本：")
    for i in range(len(texts)):
        # 排除自己（对角线 1.0），取相似度最高的那条
        best = max((j for j in range(len(texts)) if j != i), key=lambda j: sim[i][j])
        print(f"  {texts[i]}  ->  {texts[best]}（{sim[i][best]:.3f}）")


if __name__ == "__main__":
    main()
