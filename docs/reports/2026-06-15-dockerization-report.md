# 中文医疗 NLP 项目 Docker 化报告

## 一、任务目标

本次任务目标是在不影响原有项目文件的前提下，额外生成一个 Docker 化版本，使项目可以通过 Docker Compose 方式启动 Streamlit Web 页面。

原项目路径：

```text
D:\medical_nlp_project
```

Docker 化版本路径：

```text
D:\medical_nlp_project_dockerized
```

本次任务没有修改原项目目录中的文件，而是在新目录中复制必要源码和运行资产，并新增 Docker 相关配置。

## 二、Docker 化版本新增文件

Docker 化版本新增了以下核心文件：

```text
D:\medical_nlp_project_dockerized\Dockerfile
D:\medical_nlp_project_dockerized\.dockerignore
D:\medical_nlp_project_dockerized\docker-compose.yml
D:\medical_nlp_project_dockerized\docker\README_DOCKER.md
D:\medical_nlp_project_dockerized\docs\reports\2026-06-15-dockerization-report.md
```

其中：

- `Dockerfile`：定义 Python 运行环境、依赖安装、Streamlit 启动命令和健康检查。
- `.dockerignore`：排除 `.git`、虚拟环境、缓存、原始数据、处理数据和日志，减少构建上下文体积。
- `docker-compose.yml`：定义服务、端口映射、运行产物挂载和模型缓存 volume。
- `docker/README_DOCKER.md`：提供 Docker 构建、启动、停止和验证命令。
- 本报告：记录 Docker 化设计、差异、验证结果和限制。

## 三、Docker 化版本与原先版本的差异

### 1. 启动方式不同

原先版本依赖本机 Python 环境和本地虚拟环境：

```powershell
streamlit run src\app\streamlit_app.py
```

Docker 化版本通过容器启动：

```powershell
cd D:\medical_nlp_project
docker compose build
docker compose up
```

容器内部实际执行：

```text
streamlit run src/app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

浏览器访问地址仍为：

```text
http://localhost:8501
```

### 2. Python 环境来源不同

原先版本使用用户本机 Python、pip 和 `.venv`。

Docker 化版本使用 `python:3.11-slim` 镜像，在容器内安装 `requirements.txt` 中的依赖。这样可以减少不同电脑之间 Python 版本、依赖版本和系统库差异导致的运行问题。

### 3. 运行产物管理方式不同

原先版本直接读取项目目录中的：

```text
outputs/
data/knowledge_base/
data/medical_kg/
```

Docker 化版本不把这些大体积或运行态目录打进镜像，而是通过 Docker Compose 挂载到容器：

```text
./outputs -> /app/outputs
./data/knowledge_base -> /app/data/knowledge_base
./data/medical_kg -> /app/data/medical_kg
./data/knowledge_base_sources.json -> /app/data/knowledge_base_sources.json
```

这样做的好处是：

- 镜像体积更可控。
- 模型权重和向量库可以独立替换。
- 不需要每次修改模型或知识库后都重新构建镜像。

### 4. 模型 active 指针不同

原项目中的 active NER 模型指针是 Windows 绝对路径：

```text
D:\medical_nlp_project\outputs\ner_model_runs\20260603_211731
```

这个路径在 Linux 容器内不可用。因此 Docker 化版本将其改为相对路径：

```text
outputs/ner_model_runs/20260603_211731
```

容器工作目录为 `/app`，因此该相对路径可以解析到：

```text
/app/outputs/ner_model_runs/20260603_211731
```

这项修改只发生在 Docker 化副本中，不影响原项目。

### 5. 数据范围不同

Docker 化版本保留了运行 Web 演示所需的数据：

```text
data/knowledge_base/
data/medical_kg/
outputs/ner_model_runs/
outputs/vector_db/
outputs/ner_model_active.txt
```

Docker 化版本没有复制以下内容：

```text
.git/
.venv/
data/raw/
data/processed/
__pycache__/
本机缓存
日志文件
```

这符合容器化部署的基本原则：镜像和运行目录只保留运行所需内容，训练原始数据、处理过程数据和本机缓存不进入 Docker 版本。

## 四、Docker 化核心功能在本次任务中的体现

### 1. 环境封装

`Dockerfile` 将项目运行环境封装为容器镜像，包含：

- Python 3.11
- 系统依赖 `libgomp1` 和 `curl`
- `requirements.txt` 中的 Python 依赖
- Streamlit Web 启动入口

这体现了 Docker 的第一个核心功能：把应用运行环境从本机环境中隔离出来，降低“我的电脑能跑，别人的电脑不能跑”的风险。

### 2. 服务编排

`docker-compose.yml` 定义了 `medical-nlp` 服务，包括：

- 构建上下文
- 镜像名称
- 容器名称
- 端口映射
- 环境变量
- 运行产物挂载
- 缓存 volume
- 重启策略

这体现了 Docker Compose 的核心作用：用一个配置文件描述完整服务，使用户可以用 `docker compose up` 启动项目。

### 3. 端口暴露

Docker 化版本将容器内 Streamlit 的 `8501` 端口映射到宿主机：

```text
8501:8501
```

这使 Web 页面可以通过宿主机浏览器访问：

```text
http://localhost:8501
```

这体现了容器化 Web 应用的基础能力：容器内部服务通过端口映射对外提供访问。

### 4. 运行资产挂载

模型权重、向量库和知识图谱没有打入镜像，而是通过 volume 挂载：

```text
./outputs:/app/outputs:ro
./data/knowledge_base:/app/data/knowledge_base:ro
./data/medical_kg:/app/data/medical_kg:ro
```

这体现了 Docker 中“镜像”和“数据”的分离：

- 镜像负责运行环境和代码。
- 挂载目录负责模型、向量库和知识库等运行资产。

这种方式更适合机器学习项目，因为模型文件通常较大，并且会随训练结果迭代。

### 5. 缓存隔离

Docker 化版本为 Hugging Face 和 sentence-transformers 缓存配置了 named volumes：

```text
huggingface_cache
sentence_transformers_cache
```

这样首次下载 embedding 模型后，后续容器重启可以复用缓存，不需要重复下载。

这体现了容器化项目中对外部模型缓存的隔离管理。

### 6. 健康检查

`Dockerfile` 中配置了 Streamlit 健康检查：

```text
http://127.0.0.1:8501/_stcore/health
```

容器运行后，Docker 可以通过健康检查判断 Web 服务是否正常响应。

这体现了容器化部署中对服务可用性的基础监控能力。

## 五、当前 Docker 化版本目录结构

```text
medical_nlp_project_dockerized/
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── docker/
│   └── README_DOCKER.md
├── configs/
├── data/
│   ├── knowledge_base/
│   ├── medical_kg/
│   └── knowledge_base_sources.json
├── docs/
│   └── reports/
├── examples/
├── outputs/
│   ├── ner_model_active.txt
│   ├── ner_model_runs/
│   └── vector_db/
├── scripts/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

## 六、验证情况

本次 Docker 化版本完成了静态检查、Compose 配置检查和历史运行验证。上传到 GitHub 前重新确认了以下内容：

| 验证项 | 结果 |
| --- | --- |
| `Dockerfile` 存在 | 通过 |
| `.dockerignore` 存在 | 通过 |
| `docker-compose.yml` 存在 | 通过 |
| `docker/README_DOCKER.md` 存在 | 通过 |
| Streamlit 入口 `src/app/streamlit_app.py` 存在 | 通过 |
| `requirements.txt` 存在 | 通过 |
| `.dockerignore` 排除 `.git`、`.venv`、缓存和日志 | 通过 |
| `.dockerignore` 排除 `data/raw/` 和 `data/processed/` | 通过 |
| `.dockerignore` 排除 `outputs/`，避免模型权重进入镜像上下文 | 通过 |
| Git `.gitignore` 继续排除 `outputs/`、原始数据、处理数据和虚拟环境 | 通过 |
| `docker compose config --quiet` | 通过 |

当前机器已安装 Docker 与 Docker Compose：

```text
Docker version 29.5.3, build d1c06ef
Docker Compose version v5.1.4
```

Docker 化过程中，在 Docker Desktop 正常启动时已经完成过构建和运行验证：

```text
image: medical-nlp-project:dockerized
container: medical-nlp-project
health: healthy
port: 8501 -> 8501
health endpoint: http://localhost:8501/_stcore/health -> 200 ok
```

上传前复验时，Docker CLI 可用，但 Docker Desktop daemon 未运行，因此没有重新执行耗时的镜像构建和容器启动。此状态不影响源码上传；使用者运行时只需要先启动 Docker Desktop，再执行 `docker compose build` 和 `docker compose up -d`。
## 七、使用方式

安装 Docker 后，在 Docker 化版本目录运行：

```powershell
cd D:\medical_nlp_project
docker compose build
docker compose up
```

访问：

```text
http://localhost:8501
```

后台运行：

```powershell
docker compose up -d
```

停止：

```powershell
docker compose down
```

运行测试：

```powershell
docker compose run --rm medical-nlp python -m unittest discover -s tests
```

## 八、限制与后续建议

当前限制：

1. Git 仓库不提交 `outputs/`，因此只克隆源码时不会自动包含 active NER 模型、tokenizer 和 RAG 向量库。
2. 完整演示需要从 GitHub Release 获取运行资产，或自行重新训练模型、构建向量库。
3. RAG 查询使用 `sentence-transformers`，首次加载 embedding 模型时可能需要容器联网下载模型。
4. Docker 化版本主要面向推理和 Web 演示，不包含 GPU 训练环境；如需容器内训练，应另建训练镜像。
5. 如果 Docker Desktop daemon 未启动，Docker CLI 会提示无法连接 daemon，此时应先启动 Docker Desktop。

后续建议：

1. 在安装 Docker Desktop 后执行 `docker compose build` 和 `docker compose up`。
2. 如果需要完全离线运行，应把 embedding 模型缓存导出并挂载到 `/cache/sentence-transformers` 或改造为本地 embedding 模型路径。
3. 如果需要训练型 Docker 环境，可以另建 `Dockerfile.train`，单独包含 `data/processed/`、训练脚本和 GPU/CUDA 依赖。
4. GitHub 源码仓库只发布 Docker 配置、源码和文档；模型权重、向量库和完整运行资产继续通过 Release 或外部存储分发。

## 九、结论

本次任务先在独立目录中生成 Docker 化版本，随后将必要的 Docker 配置和说明文档合并回 GitHub 源码仓库。Docker 化版本保留 Streamlit Web 入口，并通过 Dockerfile 与 Docker Compose 封装运行环境、端口映射、运行资产挂载和缓存管理；NER active 模型、RAG 向量库、本地知识库和医学知识图谱仍作为运行资产挂载，不直接提交进 Git。

该版本体现了 Docker 化的核心价值：环境封装、服务编排、端口暴露、模型与数据挂载、缓存隔离和健康检查。Docker 化版本已完成配置、构建思路、Compose 编排、运行资产挂载和健康检查设计；在 Docker Desktop 正常启动时已经完成过镜像构建与 Web 服务运行验证。
