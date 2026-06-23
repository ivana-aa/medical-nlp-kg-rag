# Docker 化运行说明

本项目支持通过 Docker Compose 启动 Streamlit Web 页面。Docker 方式用于减少本机 Python、pip、虚拟环境和依赖版本差异带来的运行问题。

## 一、适用场景

适合以下用户：

- 只想快速打开 Web 页面体验 NER、RAG 和知识图谱功能。
- 不想在本机手动配置 Python 虚拟环境。
- 希望用统一环境复现项目展示效果。

注意：源码仓库不提交大型运行产物。完整演示需要同时准备 Release 包中的 `outputs/`、向量库和模型权重。

## 二、准备工作

需要先安装并启动 Docker Desktop，然后确认命令可用：

```powershell
docker --version
docker compose version
```

如果 Docker Desktop 没有启动，可能会出现无法连接 Docker daemon 的错误。这不是项目代码问题，启动 Docker Desktop 后重新执行即可。

## 三、推荐运行方式

### 方式一：使用 Release 完整演示包

从 GitHub Release 下载完整 ZIP，解压后进入项目目录：

```powershell
cd D:\medical_nlp_project
```

启动服务：

```powershell
docker compose build
docker compose up -d
```

浏览器访问：

```text
http://localhost:8501
```

停止服务：

```powershell
docker compose down
```

### 方式二：从源码仓库启动

```powershell
git clone https://github.com/ivana-aa/medical-nlp-kg-rag.git
cd medical-nlp-kg-rag
docker compose build
docker compose up -d
```

这种方式可以验证页面和代码结构，但如果没有 Release 包里的 `outputs/`，训练好的 NER 模型和已构建向量库可能不可用。需要完整展示时，请补齐 Release 运行资产。

## 四、容器入口

容器启动命令为：

```text
streamlit run src/app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

端口映射：

```text
8501:8501
```

## 五、运行产物挂载

为避免镜像体积过大，以下目录通过 Docker Compose volume 挂载到容器中：

```text
./outputs -> /app/outputs
./data/knowledge_base -> /app/data/knowledge_base
./data/medical_kg -> /app/data/medical_kg
./data/knowledge_base_sources.json -> /app/data/knowledge_base_sources.json
```

其中：

- `outputs/` 保存 active NER 模型、tokenizer 和 RAG 向量库。
- `data/knowledge_base/` 保存本地医学知识库。
- `data/medical_kg/` 保存医学知识图谱 CSV。

`outputs/ner_model_active.txt` 应使用相对路径，例如：

```text
outputs/ner_model_runs/20260603_211731
```

这样容器内 `/app` 工作目录可以正确解析 active checkpoint。

## 六、缓存策略

`sentence-transformers` 首次加载 embedding 模型时可能需要联网下载模型。Compose 文件为模型缓存配置了 Docker named volumes：

```text
huggingface_cache
sentence_transformers_cache
```

首次下载完成后，再次启动会复用缓存。

## 七、验证命令

检查 Compose 配置：

```powershell
docker compose config
```

构建镜像：

```powershell
docker compose build
```

启动容器：

```powershell
docker compose up -d
```

查看容器状态：

```powershell
docker compose ps
```

运行项目测试：

```powershell
docker compose run --rm medical-nlp python -m unittest discover -s tests
```

## 八、注意事项

- Git 仓库不提交 `.git`、`.venv`、`data/raw/`、`data/processed/`、`outputs/` 和本机缓存。
- Docker 镜像构建需要安装 Python 依赖，首次构建时间较长。
- RAG 查询如果触发 embedding 模型下载，需要容器具备网络访问能力。
- Docker 化版本主要面向推理和 Web 演示；重新训练模型建议继续使用本机 Python 环境或单独扩展训练镜像。
- 本项目仅用于 NLP 学习和工程演示，不能替代医生诊断或治疗建议。
