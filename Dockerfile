# 使用 Python 3.10 作为基础镜像（对标大纲后端环境要求）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 python 缓冲 stdout 和 stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（部分 C++ 库可能需要，如 ChromaDB 底层依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .

# 1. 强行指定双源：主源找 CPU 版 Torch，备用源（阿里云）找辅助依赖，完美绕过官方元数据错误
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --extra-index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 2. 从阿里云高速安装剩余的其他依赖
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 复制项目所有文件到工作目录
COPY . .

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]