"""
rag_builder.py - 基于 ChromaDB 的健康知识 RAG 构建与查询

用法：
  建库：python rag_builder.py --build
  查询：python rag_builder.py --query "如何科学减重"

依赖：chromadb（已安装）、sentence-transformers（需单独安装）
"""
import sys
import shutil
import os
from pathlib import Path

# 【新增】关闭 HuggingFace 联网更新检查与遥测，消除警告并提速
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 强制 HuggingFace 走国内镜像站！

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 中文轻量 Embedding 模型（首次运行自动下载，约 400MB）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ------------------------------------------------------------
# 全局缓存向量模型，避免每次查询重复加载 400MB 模型（核心提速点）
# ------------------------------------------------------------
_cached_collection = None

def get_or_create_collection():
    """获取或创建 ChromaDB collection，embedding 函数懒加载且仅加载一次"""
    global _cached_collection
    if _cached_collection is not None:
        return _cached_collection  # 如果内存里已经有了，直接秒回！

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # pyright: ignore[reportArgumentType]
        model_name=EMBEDDING_MODEL,
        device="cpu",
        normalize_embeddings=True,
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _cached_collection = client.get_or_create_collection(
        name="health_knowledge",
        embedding_function=ef,  # pyright: ignore[reportArgumentType]
    )
    return _cached_collection

# ------------------------------------------------------------
# 建库
# ------------------------------------------------------------
def build_knowledge_base():
    """读取 data/*.txt，分块 → 向量化 → 存入本地 ChromaDB"""
    if not DATA_DIR.exists():
        print("[错误] data/ 目录不存在，请先创建并放入 .txt 文件")
        return

    txt_files = list(DATA_DIR.glob("*.md"))
    if not txt_files:
        print("[错误] data/ 目录下没有 .md 文件") # 提示语也顺便改一下
        return

    print(f"发现 {len(txt_files)} 个文本文件")

    splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,
        chunk_overlap=50,
        # 新增下面这行 separators，优先按 Markdown 的双换行和单换行切分
        separators=["\n\n## ", "\n\n### ", "\n\n", "\n", " ", ""]
    )

    all_chunks = []
    for txt_file in txt_files:
        print(f"  处理: {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    print(f"共切分 {len(all_chunks)} 个文本块")

    # 清除旧向量库
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("  已清除旧向量库")

    collection = get_or_create_collection()
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(documents=all_chunks, ids=ids)

    print(f"✓ 知识库构建完成 → {CHROMA_DIR}")


# ------------------------------------------------------------
# 查询
# ------------------------------------------------------------
def query_health_knowledge(query: str) -> str:
    """根据查询返回最相关的两段文献上下文"""
    if not CHROMA_DIR.exists():
        return "[RAG 未就绪] 请先运行 python rag_builder.py --build 构建知识库"

    collection = get_or_create_collection()
    results = collection.query(query_texts=[query], n_results=2)

    # .get() 在 key 存在但值为 None 时不触发默认值，需先判空再下标
    raw_docs = results.get("documents")
    documents = raw_docs[0] if raw_docs else []
    if not documents:
        return "[RAG] 未找到相关文献"

    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"【参考资料 {i}】\n{doc}")
    return "\n\n".join(parts)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python rag_builder.py --build           # 构建知识库")
        print("  python rag_builder.py --query <问题>     # 查询知识库")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "--build":
        build_knowledge_base()
    elif cmd == "--query":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "如何健康饮食"
        print(query_health_knowledge(q))
    else:
        print(f"未知命令: {cmd}")
