# 中文医疗 NLP：实体识别、知识图谱与 RAG 问答系统

一个中文医疗自然语言处理工程示例，包含中文医疗命名实体识别、医学知识图谱构建、检索增强生成问答和 Graph-RAG 问答。项目优先使用公开数据源和本地可复现流程，不使用真实患者病历，不采集或存储患者隐私信息。

> 医疗安全声明：本项目仅用于 NLP 学习、工程验证和医学科普场景演示，不提供医疗诊断、治疗方案或用药建议。所有问答结果都应视为参考信息，不能替代医生诊断或治疗建议。如有不适或病情变化，请及时咨询正规医疗机构。

## 功能特性

- 中文医疗 NER：基于 PyTorch 和 Transformers 的 BERT token classification，支持 BIO 标注、训练、验证、测试和推理。
- 医疗实体类型：疾病、症状、药物、检查、手术、身体部位、医学检验指标。
- 本地医学知识库：支持加载 `txt`、`md`、`csv`、`json` 格式文本。
- RAG 问答：使用 `sentence-transformers` 生成向量，支持 FAISS 检索，并在 FAISS 不可用时降级为 NumPy 检索。
- Hybrid Retrieval：支持向量检索与 BM25 的混合召回。
- 医学知识图谱：支持从 CMeIE 风格关系抽取数据转换医学三元组，并用 NetworkX 进行图谱检索。
- Graph-RAG：融合文本证据和图谱三元组证据，生成带来源说明和风险提示的回答。
- Streamlit Web UI：提供项目状态、RAG 问答、NER 识别、知识图谱和 Graph-RAG 演示页面。
- 测试用例：包含 NER 后处理、CMeIE 导入、KG 过滤和 RAG 混合检索等单元测试。

## 源码仓库与 Release 包

本仓库只跟踪源码、配置、脚本、测试和小型示例数据。大型运行产物不提交到 Git，包括模型权重、向量库、原始下载数据和运行日志。

| 获取方式 | 包含内容 | 适用场景 |
| --- | --- | --- |
| GitHub source repo | 源码、配置、脚本、测试、demo 数据 | 代码阅读、二次开发、重新训练和复现实验 |
| GitHub Release package | 源码、active NER checkpoint、tokenizer、RAG 向量库、知识图谱文件 | 下载后快速运行完整本地演示 |

完整演示包：

- Release: [v1.0.0](https://github.com/ivana-aa/medical-nlp-kg-rag/releases/tag/v1.0.0)
- ZIP: [medical_nlp_project_transfer_20260611_202644.zip](https://github.com/ivana-aa/medical-nlp-kg-rag/releases/download/v1.0.0/medical_nlp_project_transfer_20260611_202644.zip)
- ZIP 大小约 366 MB，包含 active NER 模型和已构建向量库。

## 快速开始

### 方式一：运行 Release 完整演示包

适合直接体验 Web 页面和推理功能。

```powershell
cd D:\
# 解压 Release ZIP 后进入项目目录
cd D:\medical_nlp_project

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m unittest discover -s tests
streamlit run src\app\streamlit_app.py
```

浏览器打开：

```text
http://localhost:8501
```

Release 包应包含以下关键文件：

```text
outputs\ner_model_active.txt
outputs\ner_model_runs\20260603_211731\
outputs\vector_db\chunks.json
data\medical_kg\medical_kg.csv
```

### 方式二：使用 Docker 启动

适合希望减少本机 Python 环境配置的人。需要先安装并启动 Docker Desktop。

如果只克隆源码仓库，仓库中不包含大型模型权重和向量库；完整演示建议先下载 Release ZIP 并解压，使项目目录中包含 `outputs/` 和知识库运行产物。

```powershell
git clone https://github.com/ivana-aa/medical-nlp-kg-rag.git
cd medical-nlp-kg-rag

# 如果使用 Release 完整演示包，请在解压后的项目目录中执行下面命令
docker compose build
docker compose up -d
```

浏览器打开：

```text
http://localhost:8501
```

停止服务：

```powershell
docker compose down
```

Docker 说明文档见 [`docker/README_DOCKER.md`](docker/README_DOCKER.md)，面向初学者的原理说明见 [`docs/docker_for_interns.md`](docs/docker_for_interns.md)。

### 方式三：从源码复现

适合从源码重新生成 demo 数据、训练模型并构建向量库。

```powershell
git clone https://github.com/ivana-aa/medical-nlp-kg-rag.git
cd medical-nlp-kg-rag

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\prepare_ner_data.py
python scripts\train_ner.py
python scripts\evaluate_ner.py
python scripts\build_vector_db.py
streamlit run src\app\streamlit_app.py
```

Windows 环境中如果 `faiss-cpu` 安装失败，可以先安装其余依赖，项目的向量库模块会在运行时使用 NumPy 后端降级检索：

```powershell
pip install torch transformers sentence-transformers streamlit numpy pandas PyYAML seqeval tqdm scikit-learn requests networkx
```

## 数据来源与合规说明

项目只面向公开、可授权使用的医学文本或人工 demo 文本。使用者需要自行遵守各数据源的许可证、引用要求和账号注册要求。

| 数据来源 | 链接 | 项目用途 |
| --- | --- | --- |
| CBLUE 官方 GitHub | https://github.com/CBLUEbenchmark/CBLUE | 中文医疗 NLP 评测基准，包含实体识别、关系抽取、文本分类、问句匹配等任务说明 |
| CBLUE 中文 README | https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md | 查看中文任务介绍、数据集格式和使用方法 |
| CBLUE 天池入口 | https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414 | 下载或查看 CBLUE 相关中文医疗数据集，可能需要账号登录 |
| CBLUE 新闻页 | https://tianchi.aliyun.com/specials/promotion/cblue-news | 查看 CMeEE-V2、CMeIE-V2 等数据集说明 |
| OpenDataLab / OpenXLab CMeEE | https://opendatalab.com/ | 下载 CMeEE 实体识别数据，通常需要账号登录和 CLI 授权 |
| MedQuAD | https://github.com/abachaa/MedQuAD | 英文医疗问答数据，可用于 RAG 知识库和 QA 数据样例 |
| PMC Open Access Subset | https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ | 开放获取医学论文全文，可用于医学文献 RAG 和摘要任务。使用时需要检查文章许可证 |
| PubMed Central / PMC | https://pmc.ncbi.nlm.nih.gov/ | 医学论文全文和摘要来源。项目建议优先使用 PMC Open Access Subset |
| Europe PMC Open Access | https://europepmc.org/downloads/openaccess | 欧洲 PMC 开放获取文献下载入口，可作为 PMC OA 补充来源 |

明确不使用以下数据：

- 非法爬取的真实病历。
- 包含患者姓名、身份证号、手机号、住址、住院号等隐私字段的数据。
- 未确认授权范围的非开放许可全文。

## 项目结构

```text
medical_nlp_project/
├── configs/
│   └── config.yaml
├── data/
│   ├── knowledge_base/
│   ├── medical_kg/
│   ├── processed/
│   └── raw/
├── docs/
├── examples/
├── scripts/
│   ├── prepare_ner_data.py
│   ├── train_ner.py
│   ├── evaluate_ner.py
│   ├── build_vector_db.py
│   ├── import_medquad.py
│   ├── download_opendatalab_cmeee.py
│   ├── download_cmeie_from_hf.py
│   └── prepare_cmeie_kg.py
├── src/
│   ├── app/
│   ├── kg/
│   ├── ner/
│   ├── rag/
│   └── utils/
├── tests/
├── requirements.txt
└── TRANSFER_README.txt
```

`outputs/`、`data/raw/`、`data/processed/`、`.venv/` 和缓存目录默认不提交到 Git。

## 配置说明

主要配置集中在 `configs/config.yaml`：

- `ner.model_name`: 默认 `hfl/chinese-macbert-base`
- `ner.data_dir`: NER BIO 数据路径
- `ner.output_dir`: NER 模型输出目录
- `rag.embedding_model`: 默认 `shibing624/text2vec-base-chinese`
- `rag.vector_store_dir`: 向量库输出目录
- `rag.retrieval_mode`: 默认 `hybrid`
- `rag.top_k`: 默认召回证据数量

## NER 模块

NER 数据采用 BIO 格式，每行一个字符和一个标签，句子之间用空行分隔：

```text
患 O
者 O
出 O
现 O
头 B-SYMPTOM
痛 I-SYMPTOM

```

常用命令：

```powershell
python scripts\prepare_ner_data.py
python scripts\train_ner.py
python scripts\evaluate_ner.py
python scripts\evaluate_ner_lexicon.py --add-missing
python scripts\analyze_ner_errors.py
```

推理接口示例：

```python
from src.ner.predict import MedicalNERPredictor

predictor = MedicalNERPredictor("outputs/ner_model")
entities = predictor.predict("患者出现头痛和胸闷，医生建议进行血压检查，并考虑使用硝苯地平。")
print(entities)
```

Release v1.0.0 中包含一个 active CMeEE 采样训练 checkpoint。该 checkpoint 在项目测试划分上的示例指标为：

| Precision | Recall | F1 |
| --- | --- | --- |
| 0.5574 | 0.6293 | 0.5911 |

该结果用于工程流程验证，不代表临床级实体识别能力。

## CMeEE 数据流程

使用 OpenDataLab / OpenXLab 下载 CMeEE 通常需要先登录并配置访问凭证。安装额外依赖：

```powershell
pip install -r requirements-opendatalab.txt
```

下载、转换、采样、训练和评估：

```powershell
python scripts\download_opendatalab_cmeee.py --all
```

只使用本地已下载数据转换为 BIO：

```powershell
python scripts\prepare_ner_data.py --cmeee-dir data\raw\CMeEE
```

从全量 BIO 中抽样一个较小训练集：

```powershell
python scripts\sample_ner_data.py --source-dir data\processed\ner_demo --target-dir data\processed\ner_train --train 2000 --dev 500 --test 500
```

## RAG 模块

本地知识库目录：

```text
data/knowledge_base/
```

支持文件格式：

- `.txt`
- `.md`
- `.csv`
- `.json`

构建向量库：

```powershell
python scripts\build_vector_db.py
```

RAG 流程：

```text
question -> document loading -> chunking -> embedding -> vector/BM25 retrieval -> answer with evidence -> medical disclaimer
```

MedQuAD 小样本导入：

```powershell
python scripts\import_medquad.py --max-pairs 30
python scripts\build_vector_db.py
```

## 知识图谱与 Graph-RAG

项目使用 CSV 存储轻量医学知识图谱三元组，并通过 NetworkX 加载检索。CMeIE 风格数据可转换为如下格式：

```text
head,relation,tail,source
高血压,常用药物,硝苯地平,CMeIE
```

从本地 CMeIE JSON 或 JSONL 数据转换：

```powershell
python scripts\prepare_cmeie_kg.py --input data\raw\CMeIE --output data\medical_kg\medical_kg_cmeie.csv
```

构建一个更适合快速检索和人工检查的子图：

```powershell
python scripts\build_cmeie_review_kg.py --max-triples 5000
```

Graph-RAG 会同时检索：

- 文本知识库证据片段。
- 医学知识图谱三元组。
- 与问题实体相关的邻接关系。

## Web 页面

启动 Streamlit：

```powershell
streamlit run src\app\streamlit_app.py
```

或者：

```powershell
python scripts\run_app.py
```

页面包含：

- 项目总览：模型、向量库、知识库、公开数据源状态。
- 医疗问答 RAG：输入问题，展示回答、证据片段、来源和风险提示。
- 医疗 NER：输入中文医疗文本，展示实体表格、彩色标签和实体类型统计。
- 医学知识图谱与 Graph-RAG：展示三元组证据、文本证据和融合回答。

示例问题：

```text
高血压患者平时需要注意什么？
糖尿病患者为什么需要监测血糖？
肺炎有哪些症状，需要做什么检查？
What is Adult Acute Lymphoblastic Leukemia?
```

示例 NER 文本：

```text
患者出现头痛和胸闷，医生建议进行血压检查，并考虑使用硝苯地平。
```

## 测试

运行全部单元测试：

```powershell
python -m unittest discover -s tests
```

测试覆盖范围包括：

- CMeIE 数据解析和过滤。
- NER 解码、词典后处理和错误分析。
- NER predictor 后处理。
- RAG hybrid retriever。

## 运行产物管理策略

GitHub 仓库不提交大型运行产物，原因是模型权重、向量库和原始数据会显著增大仓库体积，也可能包含不同数据源的许可证约束。

默认被 `.gitignore` 排除的目录包括：

```text
outputs/
data/raw/
data/processed/
.venv/
__pycache__/
```

需要完整可运行演示时，请使用 GitHub Release 包。需要复现实验时，请从公开数据源重新下载数据并运行脚本生成产物。

## 局限性

- 示例模型和 demo 数据用于验证工程链路，不代表医学临床性能。
- 默认回答生成方式是基于检索证据的模板式生成，不接入外部大语言模型 API。
- 英文 MedQuAD 可作为 RAG 数据源，但中文医疗问答仍需要更多中文知识库数据增强。
- CMeEE、CMeIE、PMC OA 等真实数据集的使用需要遵守原始许可证、注册要求和引用要求。
- 医疗文本具有高风险属性，生产环境需要专家标注、严格评测、隐私审查和合规审核。

## 后续规划

- 引入更大规模 CMeEE/CMeIE 数据训练和系统化超参数搜索。
- 增加实体标准化，将疾病、药物、检查映射到医学术语库。
- 接入更强的中文 embedding 模型和 cross-encoder reranker。
- 支持本地大模型或可配置 LLM API 作为生成模块。
- 增加 RAG 评测集，评估召回率、答案忠实性和引用一致性。
- 扩展知识图谱 schema，增加关系置信度、来源许可证和审核状态字段。

## 许可证与引用说明

项目代码可用于学习和工程实验。第三方数据集、预训练模型和论文全文不随源码仓库再分发，使用者需要分别遵守其原始许可证、引用格式和下载平台规则。

使用公开医学数据时建议在研究报告或项目说明中列出：

- 数据集名称和版本。
- 下载入口。
- 使用任务。
- 是否经过抽样、清洗、格式转换。
- 许可证或使用条款。

## 医疗安全声明

本项目输出的所有医学回答都必须带有风险提示。系统回答仅供学习和参考，不能替代医生诊断或治疗建议。如有不适、急症或病情变化，应及时就医。
