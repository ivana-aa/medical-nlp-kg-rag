# 给新手实习生的 Docker 化完整讲解：中文医疗 NLP 项目

## 0. 这份文档解决什么问题

这份文档面向刚接触 Docker 的实习生，目标是把本项目的 Docker 化过程讲清楚：

1. 为什么要 Docker 化。
2. Docker 的核心概念是什么。
3. 本项目具体改了哪些文件。
4. 每个 Docker 文件每一部分的作用是什么。
5. 构建、启动、验证、停止分别发生了什么。
6. 为什么模型、向量库和知识库要通过 volume 挂载。
7. 遇到常见问题时如何排查。

本项目是一个中文医疗 NLP 系统，包含：

- 中文医疗 NER 实体识别。
- RAG 医疗问答。
- 医学知识图谱。
- Graph-RAG 问答。
- Streamlit Web 页面。

原始项目路径：

```text
D:\medical_nlp_project
```

Docker 化版本路径：

```text
D:\medical_nlp_project_dockerized
```

本次 Docker 化采用“独立副本”方式，没有直接修改原始项目目录。

---

## 1. 先理解一句话：Docker 到底是什么

Docker 可以理解为：

> 把一个应用运行需要的代码、依赖、系统环境和启动方式，打包成一个可重复运行的标准环境。

没有 Docker 时，项目能否运行，强依赖本机环境：

```text
你的 Python 版本
你的 pip 依赖版本
你的系统库
你的环境变量
你的模型文件位置
你的本地路径
```

这就容易出现：

```text
我电脑能跑，别人电脑不能跑。
```

有 Docker 后，运行环境被封装进镜像：

```text
代码 + Python + 依赖 + 系统库 + 启动命令 = Docker 镜像
```

别人只要安装 Docker，就可以按同样方式运行：

```powershell
docker compose up
```

---

## 2. Docker 的几个核心概念

### 2.1 镜像 Image

镜像是一个“应用运行环境模板”。

本项目构建出的镜像是：

```text
medical-nlp-project:dockerized
```

它里面包含：

- Python 3.11
- 项目源码
- `requirements.txt` 中的 Python 依赖
- Streamlit 启动命令
- 必要 Linux 系统库

可以把镜像理解成：

```text
可复制的运行环境安装包
```

### 2.2 容器 Container

容器是镜像运行起来后的进程环境。

镜像和容器的关系类似：

```text
镜像 = 类
容器 = 类创建出来的对象
```

或者：

```text
镜像 = 安装包
容器 = 正在运行的软件
```

本项目容器名是：

```text
medical-nlp-project
```

### 2.3 Dockerfile

`Dockerfile` 是构建镜像的说明书。

它告诉 Docker：

1. 用哪个基础系统。
2. 安装哪些系统依赖。
3. 安装哪些 Python 依赖。
4. 复制哪些项目代码。
5. 暴露哪个端口。
6. 容器启动时运行什么命令。

### 2.4 docker-compose.yml

`docker-compose.yml` 是服务编排文件。

它告诉 Docker Compose：

1. 服务叫什么。
2. 用哪个 Dockerfile 构建。
3. 镜像叫什么。
4. 容器叫什么。
5. 端口怎么映射。
6. 哪些目录要挂载。
7. 哪些环境变量要设置。

本项目只有一个服务：

```text
medical-nlp
```

### 2.5 Volume 挂载

volume 挂载就是把宿主机目录映射到容器目录。

例如：

```yaml
- ./outputs:/app/outputs:ro
```

含义是：

```text
宿主机当前目录下的 outputs
映射到容器内 /app/outputs
并且只读 ro
```

为什么要挂载？

因为机器学习项目里模型权重、向量库、知识库通常很大，而且会频繁替换。它们不适合直接写死进镜像。

---

## 3. 为什么本项目要 Docker 化

本项目本地运行需要：

- Python
- PyTorch
- Transformers
- sentence-transformers
- FAISS
- Streamlit
- 模型权重
- 向量库
- 本地知识库
- 医学知识图谱

这些东西在不同电脑上很容易出问题。

Docker 化后，至少能解决三类问题：

### 3.1 环境一致性

不用再关心用户本机装的是 Python 3.10、3.11 还是 3.12。

Dockerfile 里明确指定：

```dockerfile
FROM python:3.11-slim
```

也就是说，容器里固定使用 Python 3.11。

### 3.2 启动方式统一

原始项目启动：

```powershell
streamlit run src\app\streamlit_app.py
```

Docker 化后启动：

```powershell
docker compose up
```

用户不需要手动激活虚拟环境，也不需要逐个安装依赖。

### 3.3 运行资产隔离

模型和向量库仍放在项目目录里，通过 volume 挂载进入容器。

这样模型更新时，不需要重新构建镜像。

---

## 4. 本次 Docker 化新增了哪些文件

Docker 化版本中新增了这些文件：

```text
D:\medical_nlp_project_dockerized\Dockerfile
D:\medical_nlp_project_dockerized\.dockerignore
D:\medical_nlp_project_dockerized\docker-compose.yml
D:\medical_nlp_project_dockerized\docker\README_DOCKER.md
D:\medical_nlp_project_dockerized\docs\docker_for_interns.md
```

文件职责如下：

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 构建项目镜像 |
| `.dockerignore` | 控制哪些文件不进入 Docker 构建上下文 |
| `docker-compose.yml` | 定义容器服务、端口、挂载和环境变量 |
| `docker/README_DOCKER.md` | 简短运行说明 |
| `docs/docker_for_interns.md` | 本文档，详细教学说明 |

---

## 5. Dockerfile 逐行解释

当前 Dockerfile 内容：

```dockerfile
FROM python:3.11-slim
```

含义：

使用官方 Python 3.11 slim 镜像作为基础环境。

`slim` 表示精简版 Linux 系统，比完整 Debian 镜像小。

---

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers
```

这些是环境变量。

逐个解释：

| 变量 | 作用 |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1` | 不生成 `.pyc` 缓存文件 |
| `PYTHONUNBUFFERED=1` | Python 日志实时输出，方便看容器日志 |
| `PIP_NO_CACHE_DIR=1` | pip 安装后不保留缓存，减少镜像体积 |
| `STREAMLIT_SERVER_HEADLESS=true` | Streamlit 在容器里无头运行，不弹浏览器 |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` | 关闭 Streamlit 使用统计 |
| `HF_HOME=/cache/huggingface` | Hugging Face 缓存目录 |
| `SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers` | sentence-transformers 缓存目录 |

---

```dockerfile
WORKDIR /app
```

设置容器内工作目录为 `/app`。

后续所有命令默认都在 `/app` 下执行。

这很重要，因为项目代码会被复制到：

```text
/app
```

---

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
```

这一步安装 Linux 系统依赖。

| 依赖 | 作用 |
| --- | --- |
| `libgomp1` | FAISS、NumPy、scikit-learn 等库可能需要的 OpenMP 运行库 |
| `curl` | 通用网络工具，方便健康检查或调试 |

`rm -rf /var/lib/apt/lists/*` 用来删除 apt 缓存，减少镜像体积。

---

```dockerfile
COPY requirements.txt ./requirements.txt
```

把宿主机项目里的 `requirements.txt` 复制到容器 `/app/requirements.txt`。

---

```dockerfile
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt
```

升级 pip，然后安装 Python 依赖。

本项目依赖包括：

- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `networkx`

这一步耗时最长，也是镜像变大的主要原因。

---

```dockerfile
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests
COPY examples ./examples
COPY README.md ./README.md
COPY TRANSFER_README.txt ./TRANSFER_README.txt
```

复制项目源码和文档。

注意：这里没有复制 `outputs/` 和完整 `data/`。

原因是：

1. `outputs/` 包含模型权重和向量库，体积大。
2. `data/raw/`、`data/processed/` 不适合进入演示镜像。
3. 这些运行资产通过 `docker-compose.yml` 挂载。

---

```dockerfile
RUN mkdir -p /app/data /app/data/knowledge_base /app/data/medical_kg /app/outputs /cache/huggingface /cache/sentence-transformers
```

创建容器内需要的目录。

这些目录会被 volume 覆盖或用于缓存。

---

```dockerfile
COPY data/knowledge_base_sources.json ./data/knowledge_base_sources.json
```

复制一个轻量的数据来源说明文件。

---

```dockerfile
EXPOSE 8501
```

声明容器内部使用 8501 端口。

Streamlit 默认就是 8501 端口。

注意：`EXPOSE` 只是声明，真正把端口映射到宿主机的是 `docker-compose.yml`。

---

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1
```

这是健康检查。

Docker 会定期访问：

```text
http://127.0.0.1:8501/_stcore/health
```

如果返回正常，容器状态会显示：

```text
healthy
```

如果服务崩了，状态会变成：

```text
unhealthy
```

---

```dockerfile
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
```

这是容器启动时执行的命令。

拆开看：

```text
streamlit run src/app/streamlit_app.py
```

启动 Streamlit 页面。

```text
--server.address=0.0.0.0
```

允许容器外部访问。

如果写成 `127.0.0.1`，只能容器内部访问，宿主机浏览器可能打不开。

```text
--server.port=8501
```

指定端口。

---

## 6. docker-compose.yml 逐段解释

当前 compose 文件核心内容：

```yaml
services:
  medical-nlp:
```

定义一个服务，名字叫 `medical-nlp`。

---

```yaml
build:
  context: .
  dockerfile: Dockerfile
```

表示从当前目录构建镜像，使用当前目录下的 Dockerfile。

---

```yaml
image: medical-nlp-project:dockerized
container_name: medical-nlp-project
```

指定镜像名和容器名。

镜像名：

```text
medical-nlp-project:dockerized
```

容器名：

```text
medical-nlp-project
```

---

```yaml
ports:
  - "8501:8501"
```

端口映射。

格式是：

```text
宿主机端口:容器端口
```

所以：

```text
localhost:8501 -> 容器内 Streamlit 8501
```

浏览器访问：

```text
http://localhost:8501
```

---

```yaml
environment:
  STREAMLIT_SERVER_HEADLESS: "true"
  STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false"
  HF_HOME: "/cache/huggingface"
  SENTENCE_TRANSFORMERS_HOME: "/cache/sentence-transformers"
```

设置容器运行时环境变量。

这里和 Dockerfile 里的 `ENV` 有重复，是为了让运行时配置更明确。

---

```yaml
volumes:
  - ./outputs:/app/outputs:ro
  - ./data/knowledge_base:/app/data/knowledge_base:ro
  - ./data/medical_kg:/app/data/medical_kg:ro
  - ./data/knowledge_base_sources.json:/app/data/knowledge_base_sources.json:ro
```

这是最关键的部分。

它把宿主机上的运行资产挂载到容器里。

| 宿主机路径 | 容器路径 | 作用 |
| --- | --- | --- |
| `./outputs` | `/app/outputs` | NER 模型、向量库 |
| `./data/knowledge_base` | `/app/data/knowledge_base` | RAG 知识库 |
| `./data/medical_kg` | `/app/data/medical_kg` | 医学知识图谱 |
| `./data/knowledge_base_sources.json` | `/app/data/knowledge_base_sources.json` | 数据来源说明 |

最后的 `:ro` 表示 read-only，只读。

为什么只读？

因为 Docker 演示版主要做推理和展示，不希望容器误改模型或知识库。

---

```yaml
  - huggingface_cache:/cache/huggingface
  - sentence_transformers_cache:/cache/sentence-transformers
```

这是 Docker named volume。

用于保存 Hugging Face 和 sentence-transformers 模型缓存。

如果首次 RAG 查询需要下载 embedding 模型，下载结果会保存在这些 volume 中，下次启动不用重新下载。

---

```yaml
restart: unless-stopped
```

表示 Docker 会尽量自动重启容器，除非用户手动停止。

---

## 7. .dockerignore 解释

`.dockerignore` 控制哪些文件不发送给 Docker 构建过程。

如果没有 `.dockerignore`，Docker 可能会把很多无用文件都发给构建器：

- `.git`
- `.venv`
- 模型文件
- raw 数据
- 缓存
- 压缩包

这样会导致：

1. 构建很慢。
2. 镜像上下文很大。
3. 可能把隐私文件或无关文件打进去。

本项目排除了：

```text
.git
.venv
__pycache__/
data/raw/
data/processed/
outputs/
*.log
*.zip
*.rar
*.7z
```

注意：虽然 `.dockerignore` 排除了 `outputs/`，但容器运行时仍然可以用 `docker-compose.yml` 挂载 `outputs/`。

这两个阶段要分清楚：

```text
镜像构建阶段：outputs 不进镜像
容器运行阶段：outputs 挂载进容器
```

---

## 8. 为什么 outputs 不直接打进镜像

`outputs/` 包含：

```text
outputs/ner_model_runs/
outputs/vector_db/
outputs/ner_model_active.txt
```

其中 NER 模型权重几百 MB，向量库也会随着知识库增大。

如果把它们打进镜像：

1. 镜像会更大。
2. 每次换模型都要重新构建镜像。
3. GitHub / Docker 分发成本更高。
4. 不利于区分“代码版本”和“模型版本”。

所以更合理的方式是：

```text
代码和依赖 -> 镜像
模型和数据 -> volume 挂载
```

这也是机器学习项目 Docker 化的常见做法。

---

## 9. active 模型指针为什么要改

原项目里的：

```text
outputs/ner_model_active.txt
```

之前保存的是 Windows 绝对路径：

```text
D:\medical_nlp_project\outputs\ner_model_runs\20260603_211731
```

这个路径在容器里不存在。

Linux 容器里没有 `D:\` 这种盘符。

所以 Docker 化版本把它改成：

```text
outputs/ner_model_runs/20260603_211731
```

容器工作目录是：

```text
/app
```

所以相对路径会解析为：

```text
/app/outputs/ner_model_runs/20260603_211731
```

而 `/app/outputs` 又来自宿主机挂载：

```yaml
./outputs:/app/outputs:ro
```

这样模型就能在容器内被正确找到。

---

## 10. 实际运行步骤

### 10.1 检查 Docker

```powershell
docker --version
docker compose version
```

能看到版本号说明 Docker 和 Compose 可用。

### 10.2 进入 Docker 化项目目录

```powershell
cd D:\medical_nlp_project_dockerized
```

### 10.3 拉基础镜像

```powershell
docker pull python:3.11-slim
```

这一步从 Docker Hub 下载 Python 基础镜像。

如果出现：

```text
TLS handshake timeout
```

说明网络连接 Docker Hub 超时，不是 Dockerfile 写错。

### 10.4 构建项目镜像

```powershell
docker compose build
```

这一步会执行 Dockerfile。

主要过程：

1. 读取 Dockerfile。
2. 使用 `python:3.11-slim`。
3. 安装 Linux 系统依赖。
4. 安装 Python requirements。
5. 复制项目源码。
6. 生成镜像 `medical-nlp-project:dockerized`。

本项目构建出的镜像约：

```text
DISK USAGE: 9.11GB
CONTENT SIZE: 3.1GB
```

### 10.5 启动服务

后台启动：

```powershell
docker compose up -d
```

前台启动并看日志：

```powershell
docker compose up
```

启动后访问：

```text
http://localhost:8501
```

### 10.6 查看状态

```powershell
docker compose ps
```

如果正常，会看到类似：

```text
medical-nlp-project   Up   healthy   0.0.0.0:8501->8501/tcp
```

### 10.7 查看日志

```powershell
docker logs --tail 80 medical-nlp-project
```

正常日志会包含：

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### 10.8 健康检查

PowerShell 访问：

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8501/_stcore/health"
```

正常返回：

```text
200 ok
```

### 10.9 进入容器检查文件

```powershell
docker compose exec -T medical-nlp python -c "from pathlib import Path; print(Path('outputs/ner_model_active.txt').read_text())"
```

如果能输出：

```text
outputs/ner_model_runs/20260603_211731
```

说明模型指针在容器内可见。

### 10.10 停止服务

```powershell
docker compose down
```

这会停止并删除容器，但不会删除镜像和 volume。

---

## 11. 这次项目做到哪一步了

本项目 Docker 化已经完成过以下验证：

1. Docker 可用。
2. `python:3.11-slim` 基础镜像已成功拉取。
3. `docker compose build` 已成功构建镜像。
4. 镜像名为 `medical-nlp-project:dockerized`。
5. 容器曾成功启动。
6. 容器状态曾达到 `healthy`。
7. `http://localhost:8501/_stcore/health` 曾返回 `200 ok`。
8. 容器内能看到：
   - `src/app/streamlit_app.py`
   - `outputs/ner_model_active.txt`
   - `outputs/ner_model_runs/20260603_211731`
   - `outputs/vector_db/chunks.json`
   - `data/knowledge_base`
   - `data/medical_kg`

如果当前 `docker compose ps` 没有容器，说明服务现在只是停止了，重新执行：

```powershell
cd D:\medical_nlp_project_dockerized
docker compose up -d
```

即可重新启动。

---

## 12. 为什么镜像这么大

构建出的镜像约：

```text
9.11GB
```

主要原因不是项目代码大，而是深度学习依赖大。

尤其是：

```text
torch
transformers
sentence-transformers
faiss-cpu
scipy
pyarrow
```

其中 PyTorch 在 Linux 下可能拉取较多运行库，包括 CUDA 相关包。

所以会看到镜像明显大于普通 Web 项目。

普通 Flask / FastAPI 项目可能几百 MB。

深度学习项目 Docker 镜像几 GB 是常见现象。

如果要优化，可以做：

1. 使用 CPU-only PyTorch。
2. 单独写 Docker 专用 requirements。
3. 不安装训练相关依赖，只保留推理相关依赖。
4. 使用多阶段构建。
5. 把模型和缓存继续放 volume，不打进镜像。

---

## 13. 常见问题排查

### 13.1 Docker Desktop 一直 Starting

可能原因：

- WSL2 未启动。
- Docker Desktop 第一次启动较慢。
- WSL 后端异常。

处理：

```powershell
wsl --shutdown
```

然后重启 Docker Desktop。

必要时重启电脑。

### 13.2 拉镜像超时

报错类似：

```text
TLS handshake timeout
```

说明连接 Docker Hub 超时。

处理：

```powershell
docker pull python:3.11-slim
```

如果仍失败：

- 换网络。
- 使用代理。
- 配置 Docker 镜像加速器。

### 13.3 C 盘空间不足

Docker Desktop 默认把镜像、容器、构建缓存放在 C 盘。

查看 Docker 占用：

```powershell
docker system df
```

清理构建缓存：

```powershell
docker builder prune
```

注意：

不要一上来就执行：

```powershell
docker system prune -a
```

这个会更激进，可能删除刚构建好的镜像。

### 13.4 端口被占用

如果 8501 被占用，会启动失败。

可以改 `docker-compose.yml`：

```yaml
ports:
  - "8502:8501"
```

然后访问：

```text
http://localhost:8502
```

### 13.5 页面能打开但 RAG 很慢

可能是首次加载 embedding 模型。

`sentence-transformers` 可能需要下载：

```text
shibing624/text2vec-base-chinese
```

下载后会缓存到：

```text
/cache/sentence-transformers
```

这个目录由 Docker volume 保存。

### 13.6 NER 模型找不到

重点检查：

```text
outputs/ner_model_active.txt
outputs/ner_model_runs/20260603_211731
```

以及 compose 挂载：

```yaml
- ./outputs:/app/outputs:ro
```

容器内检查：

```powershell
docker compose exec -T medical-nlp python -c "from pathlib import Path; print(Path('outputs/ner_model_active.txt').read_text())"
```

---

## 14. 原版本和 Docker 版本的区别

| 对比项 | 原版本 | Docker 化版本 |
| --- | --- | --- |
| 运行环境 | 本机 Python / venv | 容器内 Python 3.11 |
| 启动方式 | `streamlit run ...` | `docker compose up` |
| 依赖安装 | 手动 `pip install` | 构建镜像时自动安装 |
| 模型路径 | Windows 本地路径 | 容器内 `/app/outputs` 挂载 |
| active 指针 | 原 Windows 绝对路径 | Docker 副本中改为相对路径 |
| 知识库 | 本地读取 | volume 挂载只读读取 |
| 向量库 | 本地读取 | volume 挂载只读读取 |
| 可移植性 | 依赖本机环境 | 更容易迁移到其他装有 Docker 的机器 |
| 镜像体积 | 无镜像 | 约 9GB |

---

## 15. 面试或汇报时怎么讲

可以这样说：

> 我将原本依赖本机 Python 环境运行的中文医疗 NLP 项目 Docker 化，使用 `python:3.11-slim` 作为基础镜像，通过 Dockerfile 安装 PyTorch、Transformers、sentence-transformers、FAISS 和 Streamlit 等依赖，并用 Docker Compose 统一管理端口映射、环境变量和运行资产挂载。

继续补充：

> 由于模型权重、RAG 向量库和医学知识库体积较大，我没有把它们直接打进镜像，而是通过 volume 挂载到容器内 `/app/outputs`、`/app/data/knowledge_base` 和 `/app/data/medical_kg`，实现代码环境和运行数据分离。

再强调验证：

> 最终完成了镜像构建、容器启动、Streamlit 健康检查和容器内路径验证，服务可以通过 `http://localhost:8501` 访问。

---

## 16. 一个新手需要真正理解的重点

不要只背命令，要理解这几件事：

1. Dockerfile 负责“怎么做镜像”。
2. docker-compose.yml 负责“怎么运行服务”。
3. 镜像是模板，容器是运行实例。
4. 端口映射让浏览器能访问容器里的 Web 服务。
5. volume 挂载让容器能读取宿主机的大模型和数据。
6. `.dockerignore` 是为了避免把无关文件送进构建过程。
7. 机器学习项目镜像大是正常现象，但可以通过 CPU-only 依赖进一步优化。
8. Docker 化不是把所有东西塞进镜像，而是合理拆分代码、环境、模型和数据。

---

## 17. 推荐练习

为了真正掌握本项目 Docker 化，实习生可以按以下顺序练习：

1. 运行：

```powershell
docker --version
docker compose version
```

2. 进入项目：

```powershell
cd D:\medical_nlp_project_dockerized
```

3. 启动：

```powershell
docker compose up -d
```

4. 查看状态：

```powershell
docker compose ps
```

5. 打开页面：

```text
http://localhost:8501
```

6. 查看日志：

```powershell
docker logs --tail 80 medical-nlp-project
```

7. 检查容器文件：

```powershell
docker compose exec -T medical-nlp python -c "from pathlib import Path; print(Path('outputs/vector_db/chunks.json').exists())"
```

8. 停止：

```powershell
docker compose down
```

完成这些练习后，基本就掌握了本项目 Docker 化的核心流程。

---

## 18. 总结

本项目 Docker 化的本质是：

```text
把中文医疗 NLP 项目的运行环境封装进镜像，
把模型、向量库和知识库作为运行资产挂载进容器，
再通过 Streamlit 暴露 Web 页面。
```

最终形成的运行链路是：

```text
docker compose up
        ↓
启动 medical-nlp 容器
        ↓
容器运行 Streamlit
        ↓
读取 /app/outputs 中的 NER 模型和向量库
        ↓
读取 /app/data 中的知识库和知识图谱
        ↓
浏览器访问 http://localhost:8501
```

这就是本项目 Docker 化的完整实现逻辑。
