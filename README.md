# 基于中文医疗文本的实体识别、医学知识图谱与 RAG 医疗问答系统

这是一个面向医疗智能体原型的中文医疗 NLP 项目，整合中文医疗 NER、医学知识图谱、RAG 与 Graph-RAG 问答能力，适合课程报告、简历展示和后续工程扩展。

1. 中文医疗命名实体识别 NER：从医疗文本中识别疾病、症状、药物、检查、手术、身体部位、检验指标等实体。
2. 医学知识图谱 KG：使用 CSV + NetworkX 构建轻量图谱，支持实体查询、关系筛选和 NER 辅助候选关系生成。
3. 医疗文本 RAG / Graph-RAG 问答：从本地医学知识库和医学图谱中检索证据，并生成带引用片段、结构化三元组和风险提示的回答。

核心流程：

```text
医学文本 → 数据清洗 → BIO 标注 → MacBERT NER → 实体识别
      → 医学知识图谱 → FAISS 向量检索 → Graph-RAG 问答 → Streamlit 展示
```

项目默认不使用真实患者病历，不采集隐私数据，不进行非法爬取。内置的小型 demo 数据是人工编写的医学科普风格文本，仅用于跑通流程。

## 项目结构

```text
medical_nlp_project/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── knowledge_base/
│   └── medical_kg/
├── src/
│   ├── ner/
│   ├── kg/
│   ├── rag/
│   ├── app/
│   └── utils/
├── scripts/
├── examples/
└── outputs/
```

## 数据来源说明

推荐使用公开、可靠、合规的数据来源。不要使用非法爬取的真实病历，不要使用包含患者隐私的数据；如果数据集页面要求登录、注册、申请或遵守特定许可证，请按原始平台规则使用。除原始发布方页面外，也可以使用 OpenDataLab 等公开数据平台下载或管理数据集，但需要核对数据集来源、版本、许可证和使用条款，不把第三方镜像简单等同于原始发布方授权。

| 数据来源 | 用途 | 使用说明 |
| --- | --- | --- |
| [CBLUE 官方 GitHub](https://github.com/CBLUEbenchmark/CBLUE) | 中文医疗 NLP 评测基准，包含中文医学实体识别、关系抽取、文本分类、问句匹配等任务。 | 适合本项目的 NER、关系抽取和后续医学文本分类扩展。CMeEE 可用于实体识别，CMeIE 可用于关系抽取或知识图谱构建。 |
| [CBLUE 中文 README](https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md) | 查看中文说明、任务介绍、数据集使用方法。 | README 中给出了任务介绍、数据目录格式、训练样例和 CMeEE/CMeIE 等任务说明，适合作为接入真实数据前的说明文档。 |
| [CBLUE 天池数据集入口](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414) | 下载或查看 CBLUE 相关中文医疗数据集。 | 天池页面可能需要登录天池/阿里云账号后下载，使用时应遵守页面的数据集协议和竞赛/评测要求。 |
| [OpenDataLab](https://opendatalab.com/) | 公开数据平台，可用于检索、下载或管理公开数据集。 | 如果平台提供 CBLUE/CMeEE/CMeIE 等数据集入口，可以作为数据下载方式之一；使用前需要核对数据集版本、原始来源、许可证和平台使用条款。 |
| [CBLUE 新闻页](https://tianchi.aliyun.com/specials/promotion/cblue-news) | 查看 CMeEE-V2、CMeIE-V2 等数据集说明。 | 适合跟踪 CBLUE 2.0/3.0 数据集更新，如 CMeEE-V2、CMeIE-V2、CMedCausal、IMCS-V2 等。 |
| [MedQuAD 医疗问答数据集](https://github.com/abachaa/MedQuAD) | 英文医疗问答数据集，适合医疗 QA、RAG、问答系统微调。 | 数据集包含来自 NIH 相关网站的医学问答对，可用于扩展英文 RAG 知识库；使用时注意仓库许可证和原始来源版权说明。 |
| [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | 开放获取医学论文全文，适合医学文献 RAG、摘要生成、医学知识库构建。 | 优先使用 OA 子集，不要批量抓取非开放许可全文；每篇文章的许可证可能不同，需要保留来源和许可证信息。 |
| [PubMed Central / PMC](https://pmc.ncbi.nlm.nih.gov/) | 医学论文全文和摘要来源。 | 项目中优先使用 PMC Open Access Subset；PMC 主站可用于检索和查看文献，不要随意抓取非开放许可全文。 |
| [Europe PMC Open Access](https://europepmc.org/downloads/openaccess) | 欧洲 PMC 的开放获取文献下载入口，可作为 PMC OA 的补充数据来源。 | 可通过 Web services、OAI 或 FTP 获取开放获取子集；使用时需要检查每篇文章的具体许可证条款。 |

本项目内置 demo 数据：

- NER demo：运行 `python scripts/prepare_ner_data.py` 后生成到 `data/processed/ner_demo/`。
- RAG demo 知识库：运行同一脚本后生成到 `data/knowledge_base/`，包含 `md/json/csv` 三种格式示例。

即使暂时没有下载真实公开数据，也可以先使用内置 demo 数据跑通完整流程；真实数据接入后，再替换或扩展 `data/raw/` 与 `data/knowledge_base/` 中的内容。

## 环境安装

建议使用 Python 3.10 到 3.12，并在虚拟环境中安装依赖：

```bash
cd medical_nlp_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果 Windows 中提示 `pip` 不存在，可以尝试：

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 `faiss-cpu` 在 Windows 环境安装失败，项目会在构建向量库时自动退回到 NumPy 点积检索，保证 demo 可运行。正式实验或简历展示建议在 Linux、WSL 或 Conda 环境中安装 FAISS。

## GitHub 版本说明

GitHub 仓库默认只包含源码、配置、README、测试、demo 知识库和轻量医学知识图谱，不包含以下本地生成或大体积文件：

- `outputs/`：NER 模型权重、评估报告、向量库、日志等。
- `data/raw/`：原始下载数据。
- `data/processed/`：转换后的 BIO 训练数据。
- `.venv/`、缓存和临时文件。

这样做是为了避免把大模型权重和本机缓存提交到 GitHub。当前 active NER 模型权重约 388MB，普通 GitHub 仓库不适合直接提交。

因此，其他人使用本仓库有两种方式：

1. **源码复现实验**：clone 仓库后按 README 运行 `prepare_ner_data.py`、`train_ner.py`、`evaluate_ner.py`、`build_vector_db.py`，重新生成本地模型和向量库。
2. **完整演示包**：从 GitHub Release 下载包含 `outputs/` 的迁移压缩包，解压后按 `TRANSFER_README.txt` 创建虚拟环境并安装依赖，即可加载已训练模型和向量库进行展示。

## 数据准备

生成内置 demo 数据：

```bash
python scripts/prepare_ner_data.py
```

可选：使用已经下载好的 CBLUE/CMeEE 数据。数据可以来自 CBLUE 原始发布入口、天池数据集页、OpenDataLab 等公开数据平台；无论从哪里下载，都应确认其来源、版本和许可证。将文件放到：

```text
data/raw/CMeEE/
├── CMeEE_train.json
├── CMeEE_dev.json
└── CMeEE_test.json
```

然后运行：

```bash
python scripts/prepare_ner_data.py --cmeee-dir data/raw/CMeEE
```

说明：CMeEE 存在嵌套实体，普通 BIO 序列标注无法完整表达重叠实体。当前转换脚本会保留第一个非重叠实体，适合初学者先跑通 BERT token classification 流程。后续可改为 span classification 或 GlobalPointer。

### 使用 OpenDataLab / OpenXLab 下载 CMeEE

OpenDataLab 的旧版 `odl` 命令已提示弃用，当前推荐使用新版 `openxlab` CLI。经本项目验证，`OpenDataLab/CMeEE` 仓库可查看到 `/raw/CMeEE.tar.gz` 文件；下载时平台要求先登录并配置个人 AK/SK，这是正常的数据平台鉴权流程。

安装下载工具：

```bash
py -m pip install -r requirements-opendatalab.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

登录 OpenXLab / OpenDataLab 账号：

```bash
openxlab login
```

如果 Windows 提示找不到 `openxlab`，可以使用 Python Scripts 目录中的可执行文件，例如：

```bash
"C:\Users\<你的用户名>\AppData\Local\Programs\Python\Python312\Scripts\openxlab.exe" login
```

登录后可直接运行本项目提供的一键脚本：

```bash
py scripts/download_opendatalab_cmeee.py --all
```

该脚本会执行以下步骤：

1. 从 `OpenDataLab/CMeEE` 下载 `/raw/CMeEE.tar.gz` 到 `data/raw/CMeEE/`。
2. 解压压缩包。
3. 调用 `scripts/prepare_ner_data.py --cmeee-dir data/raw/CMeEE` 转换为 BIO。
4. 调用 `scripts/sample_ner_data.py` 生成 CPU 友好的真实数据训练子集。
5. 调用 `scripts/train_ner.py` 重新训练 NER。
6. 调用 `scripts/evaluate_ner.py` 生成评估报告。

如果你已经手动下载了压缩包，也可以跳过下载：

```bash
py scripts/download_opendatalab_cmeee.py --skip-download --all
```

如果是 CPU 电脑，建议先用真实 CMeEE 的小样本训练展示版：

```bash
py scripts/download_opendatalab_cmeee.py --skip-download --prepare
py scripts/sample_ner_data.py --train 500 --dev 150 --test 150
py scripts/train_ner.py
py scripts/evaluate_ner.py
```

当前 `configs/config.yaml` 默认读取 `data/processed/ner_train/`。内置 demo 模式下，`prepare_ner_data.py` 会把 demo BIO 同步写入这个目录；接入 CMeEE 后，`sample_ner_data.py` 会把真实 CMeEE 子集写入这个目录。全量 BIO 文件仍保存在 `data/processed/ner_demo/`，如果要全量训练，可把配置中的 `ner.data_dir` 改回 `data/processed/ner_demo`。

## NER 模型训练

默认模型为 `hfl/chinese-macbert-base`，可在 `configs/config.yaml` 中替换为：

- `bert-base-chinese`
- `hfl/chinese-roberta-wwm-ext`
- `hfl/chinese-macbert-base`

训练：

```bash
python scripts/train_ner.py
```

模型和 tokenizer 会保存到：

```text
outputs/ner_model/
```

评估测试集：

```bash
python scripts/evaluate_ner.py
```

评估开发集：

```bash
python scripts/evaluate_ner.py --split dev
```

指标包括 Precision、Recall、F1 和 `seqeval` 分类报告。

NER 错误分析：

```bash
python scripts/analyze_ner_errors.py --split test --batch-size 16
```

该脚本会输出 `TP (True Positive，真正例)`、`FP (False Positive，假阳性/误报)`、`FN (False Negative，假阴性/漏报)`，并统计 `Boundary Error (边界错误)`、`Label Confusion (类别混淆)`、高频误报实体和高频漏报实体。报告默认保存到 `outputs/logs/ner_error_analysis_*.json`，用于判断下一步应优先做数据清洗、阈值后处理、CRF 或 GlobalPointer/span NER。

NER 词典后处理实验：

```bash
python scripts/evaluate_ner_lexicon.py --split test --batch-size 16 --min-count 5
```

`Gazetteer (实体词典)` 会从训练集 BIO 标注中统计高频实体文本及其多数类别，用于可选的 `Label Correction (标签纠正)`。当前实验只建议使用类别纠正，不建议打开 `--add-missing`，因为自动补漏会提高 Recall 但明显降低 Precision。该实验不会修改 active 模型，也不会覆盖评估报告。Streamlit 的 NER 推理已接入该标签纠正能力，并继续保留小型 demo 医学词表作为展示兜底。

当前项目已经接入 OpenDataLab CMeEE，并在 CPU 环境下使用 `2000/500/500` 的真实 CMeEE 中等子集完成训练。最近一次本地测试集结果示例：

```text
Precision: 0.4717
Recall:    0.5799
F1:        0.5203
```

该结果基于 `data/processed/ner_train/` 中的 CMeEE 小样本，适合 CPU 电脑先完成简历展示版流程；如果要追求正式效果，建议使用 GPU 对 `data/processed/ner_demo/` 中的全量 CMeEE BIO 数据重新训练并调参。

## 医学知识图谱与 Graph-RAG

项目新增轻量医学知识图谱模块，默认使用 CSV + NetworkX，不依赖 Neo4j，适合 Windows 普通电脑本地运行。

图谱数据文件：

```text
data/medical_kg/medical_kg.csv
```

CSV 字段：

```text
head,relation,tail,head_type,tail_type,source
```

内置 demo 三元组覆盖肺炎、糖尿病、高血压、常见症状、推荐检查、就诊科室、相关药物和用药风险。例如：

```text
肺炎 - 常见症状 - 咳嗽
肺炎 - 推荐检查 - 胸部CT
糖尿病 - 推荐检查 - 糖化血红蛋白
高血压 - 就诊科室 - 心血管内科
二甲双胍 - 用药风险 - 胃肠道反应
```

新增代码模块：

```text
src/kg/
├── kg_loader.py    # CSV 加载、NetworkX 图构建、实体/关系统计
├── kg_query.py     # 实体查询、关系筛选、问题实体匹配
├── kg_builder.py   # NER 辅助候选关系生成
├── cmeie_importer.py # CMeIE 关系抽取数据转 KG CSV
└── graph_rag.py    # 图谱证据 + FAISS 文本证据融合回答
```

Graph-RAG 流程：

```text
用户问题 → NER 或关键词匹配实体 → 图谱检索三元组
      → FAISS 检索文本证据 → 融合证据 → 模板式回答 → 医疗风险提示
```

示例问题：

```text
肺炎有哪些症状，需要做什么检查？
```

示例图谱证据：

```text
肺炎 - 常见症状 - 咳嗽
肺炎 - 常见症状 - 发热
肺炎 - 推荐检查 - 胸部CT
肺炎 - 推荐检查 - 血常规
```

NER 辅助图谱构建功能会从输入文本中识别实体，并使用透明规则生成候选关系：

- 疾病 + 症状 + “表现为、症状、伴有、出现、常见”等触发词 → `疾病 - 可能症状 - 症状`
- 疾病 + 检查 + “检查、建议、可行、推荐”等触发词 → `疾病 - 可能检查 - 检查`
- 疾病 + 科室 + “就诊、挂号、科室、建议”等触发词 → `疾病 - 可能科室 - 科室`

这些候选关系只在页面展示，默认不写入主图谱，避免污染 demo 数据。CMeEE/NER 标签与图谱类型不完全一致时，通过 `ENTITY_TYPE_MAPPING` 做类型映射，例如 `DISEASE → 疾病`、`SYMPTOM → 症状`、`EXAM/TEST → 检查`。

### CMeIE 关系数据导入

`CMeIE (Chinese Medical Information Extraction，中文医学信息抽取)` 是 CBLUE 中的医学关系抽取任务。它提供 `SPO (Subject-Predicate-Object，主语-谓词-宾语三元组)` 标注，可用于扩展 `KG (Knowledge Graph，知识图谱)`。

项目已提供独立转换脚本，可把 CMeIE 的 `spo_list` 转成现有 CSV 图谱格式：

```powershell
python scripts/prepare_cmeie_kg.py --input data/raw/CMeIE --output data/medical_kg/medical_kg_cmeie.csv
```

也可以直接从 Hugging Face 数据集仓库 `Aunderline/CMeIE` 下载 `53_schemas.jsonl`、`CMeIE_train.jsonl`、`CMeIE_dev.jsonl`、`CMeIE_test.jsonl`，并在下载后转换为独立 KG CSV：

```powershell
python scripts/download_cmeie_from_hf.py --convert
```

默认输出为 `data/medical_kg/medical_kg_cmeie.csv`，不会覆盖主图谱 `data/medical_kg/medical_kg.csv`。建议先生成审核版子图谱，再人工抽样确认：

```powershell
python scripts/build_cmeie_review_kg.py
```

该脚本会把 CMeIE 原始关系映射到项目展示更容易理解的关系类型，例如 `临床表现 → 常见症状`、`影像学检查/实验室检查 → 推荐检查`、`药物治疗 → 相关药物`，并过滤空实体、过长实体、Markdown/URL 噪声和超出上限的高频关系。默认输出为 `data/medical_kg/medical_kg_cmeie_reviewed.csv`，统计报告保存到 `outputs/logs/cmeie_review_kg_report.json`。审核版仍不会自动合并进主图谱。

## RAG 知识库构建

本项目支持加载以下格式：

- `.txt`
- `.md`
- `.csv`
- `.json`

将医学文本放入：

```text
data/knowledge_base/
```

可选：从公开 GitHub 仓库导入一份可追溯的 MedQuAD 医疗问答样本，用于扩容 RAG 知识库。脚本会缓存原始 XML 到 `data/raw/MedQuAD/`，并生成带来源链接、许可证和数据集名称的 JSON 文件：

```bash
python scripts/import_medquad.py --max-pairs 48 --max-pairs-per-folder 8 --max-files-per-folder 4
```

如果需要继续扩容，可以调大 `--max-pairs`、`--max-pairs-per-folder` 和 `--max-files-per-folder`。建议每次扩容后重新运行向量库构建脚本。

构建向量库：

```bash
python scripts/build_vector_db.py
```

默认 embedding 模型为 `shibing624/text2vec-base-chinese`，向量库保存到：

```text
outputs/vector_db/
```

RAG 回答默认使用“混合检索证据 + 模板生成”，不依赖外部大模型 API。检索层支持：

- `Vector Retrieval`：基于 `sentence-transformers` embedding 和 FAISS/NumPy 的语义向量检索。
- `BM25`：轻量关键词检索，对疾病、药物、检查等精确医学词更敏感。
- `RRF`：使用 Reciprocal Rank Fusion 融合向量检索和 BM25 排名。

所有回答都会附带：

```text
风险提示：本回答仅供学习和参考，不能替代医生诊断或治疗建议；如有不适或病情变化，请及时到正规医疗机构就诊。
```

当前知识库已经包含两类内容：

- 人工编写的中文 demo 医学科普文本，用于离线演示常见病问答。
- 公开来源增强条目，保存在 `data/knowledge_base/public_medical_sources.json`，包含 MedQuAD、PMC Open Access Subset、Europe PMC Open Access 的来源说明和中文改写示例。
- MedQuAD 真实问答样本，保存在 `data/knowledge_base/medquad_qa_sample.json`，当前导入 48 条英文医疗问答对，适合展示 RAG 的公开数据支撑和证据引用能力。

来源元数据保存在：

```text
data/knowledge_base_sources.json
```

构建向量库后，Streamlit 的 RAG 证据卡片会显示：

- 本地文件来源
- 数据集名称
- 来源链接
- 许可证或许可证确认方式
- 证据片段和相似度分数

最近一次构建结果：

```text
Loaded documents: 62
Generated chunks: 142
Vector backend: faiss
```

## Web 演示运行

按顺序运行：

```bash
python scripts/prepare_ner_data.py
python scripts/train_ner.py
python scripts/evaluate_ner.py
python scripts/build_vector_db.py
streamlit run src/app/streamlit_app.py
```

也可以使用：

```bash
python scripts/run_app.py
```

打开 Streamlit 页面后，可以测试：

- 医疗问答 RAG：输入医学问题，查看回答、证据片段和来源。
- 医疗 NER：输入中文医疗文本，查看识别出的实体、类型和字符位置。
- 医学知识图谱与 Graph-RAG：查询图谱三元组，运行“肺炎有哪些症状，需要做什么检查？”等 Graph-RAG 问题，并查看图谱证据、文本证据和风险提示。
- NER 辅助图谱构建：输入医学文本，查看 NER 实体和候选关系；候选关系默认不写入主图谱。
- 项目总览：查看模型状态、向量库状态、知识库规模、NER 评估指标和数据合规说明。

## 项目效果展示

NER 输入示例：

```text
患者出现头痛和胸闷，医生建议进行血压检查，并考虑使用硝苯地平。
```

可能输出：

```text
头痛 / SYMPTOM
胸闷 / SYMPTOM
血压检查 / EXAM
硝苯地平 / DRUG
```

RAG 输入示例：

```text
高血压患者平时需要注意什么？
```

系统会检索本地知识库中与高血压相关的片段，生成带 `[1]`、`[2]` 引用编号的回答，并展示每条证据的来源文件和相似度分数。

Graph-RAG 输入示例：

```text
肺炎有哪些症状，需要做什么检查？
```

系统会先匹配“肺炎”等图谱实体，检索结构化三元组，再融合 FAISS 文本证据生成回答。页面会分别展示：

- 识别/匹配实体
- 图谱结构化证据
- 文本检索证据
- 最终回答和医疗风险提示

注意：demo 数据很小，训练 1 个 epoch 只是为了跑通流程，不代表真实模型效果。

## 答辩演示路线

建议按下面顺序演示，3 到 5 分钟内可以讲清楚项目价值：

1. 打开 Streamlit 的“项目总览”页，先说明这是一个包含 NER、RAG、医学知识图谱和 Graph-RAG 的中文医疗 NLP 项目，并展示从数据、NER、知识库、检索到安全回答的完整流程。
2. 指出侧边栏中的运行状态：NER 模型已就绪、向量库已就绪、知识库文件数、检索分块数和向量后端。
3. 如需说明实验完整性，展开 NER 评估指标区：`Precision / Recall / F1` 来自 `py scripts/evaluate_ner.py` 生成的 active checkpoint `eval_report.json`。
4. 切到“医疗问答 RAG”，输入 `高血压患者平时需要注意什么？`，重点展示回答中有引用编号、下方有证据片段、来源文件和相似度。
5. 切到“医疗 NER”，输入 `患者出现头痛和胸闷，医生建议进行血压检查，并考虑使用硝苯地平。`，展示实体表格、彩色标签和实体类型统计。
6. 切到“医学知识图谱与 Graph-RAG”，输入 `肺炎有哪些症状，需要做什么检查？`，展示图谱三元组、文本证据和融合回答。
7. 在“NER 辅助图谱构建”中输入 `肺炎患者常见症状表现为咳嗽和发热，建议进行胸部CT和血常规检查，并到呼吸内科就诊。`，展示候选关系生成。
8. 最后强调医疗安全设计：项目使用公开或人工 demo 数据，所有问答都带风险提示，不能替代医生诊断。

说明：内置 demo 数据只用于首次跑通工程链路。当前版本已接入 OpenDataLab CMeEE，并训练了真实数据子集模型；为了让报告演示稳定，Streamlit 的 NER 演示采用“BERT-NER 模型输出 + 训练集实体词典标签纠正 + 医学词表补充”的方式展示，正式效果仍应以 `py scripts/evaluate_ner.py` 和 `py scripts/evaluate_ner_lexicon.py` 的评估结果为准。

## 当前优化进展

- 已通过 OpenDataLab/OpenXLab 下载 CMeEE，原始压缩包保存在 `data/raw/CMeEE/`。
- 已将 CMeEE 转换为 BIO：全量文件保存在 `data/processed/ner_demo/`，CPU 友好训练子集保存在 `data/processed/ner_train/`。
- 训练配置使用 `hfl/chinese-macbert-base`、学习率 `5e-5`，最近一次 CMeEE 中等子集测试 F1 为 `0.5203`。
- 模型保存改为 `outputs/ner_model_runs/<run_id>` + `outputs/ner_model_active.txt` 指针，避免 Streamlit 页面打开时 Windows 文件锁导致训练保存失败。
- Streamlit 页面会读取 active 模型目录和最新 `eval_report.json`，不需要手动改路径。
- 已新增 CSV + NetworkX 轻量医学知识图谱，内置肺炎、糖尿病、高血压、药物风险等 demo 三元组。
- 已新增 Graph-RAG 问答链路：实体匹配、图谱三元组检索、FAISS 文本证据检索和模板式融合回答。
- 已新增 NER 辅助图谱构建演示：从文本中识别实体，并用规则生成“可能症状/可能检查/可能科室”候选关系。

## 截图建议

如果要放到课程报告、简历附件或作品集中，建议保存三张图：

- 项目总览页：突出模块完整性、流程图、状态面板和数据合规说明。
- RAG 问答页：突出“问题、回答、证据来源、相似度、免责声明”。
- NER 识别页：突出“输入文本、彩色实体标签、实体明细表、类型统计”。
- 医学知识图谱与 Graph-RAG 页：突出“实体查询、图谱证据、文本证据、最终回答、候选关系”。

截图时可以使用浏览器缩放 90% 到 100%，并让侧边栏保持展开，这样更像完整工程项目，而不是单个脚本 demo。

## 可写进简历的项目描述

> 基于 PyTorch、Transformers、sentence-transformers、FAISS、NetworkX 和 Streamlit 构建中文医疗 NLP 原型系统，实现医疗命名实体识别、医学知识图谱、RAG 与 Graph-RAG 医疗问答。NER 模块支持 BIO 数据处理、MacBERT token classification 训练评估和实体抽取推理；RAG 模块支持多格式医学文本加载、分块、向量化、相似度检索和证据引用；知识图谱模块基于 CSV + NetworkX 实现实体查询、关系筛选、NER 辅助候选关系生成和图谱证据融合。项目遵循公开数据和隐私合规原则，所有医疗回答均附带风险提示。

岗位能力映射：

- Python / PyTorch：数据处理、模型训练、推理封装和模块化工程实现。
- Transformer / 医疗 NER：MacBERT token classification、BIO 标注、CMeEE 数据转换与评估。
- 医疗文本处理：中文医学实体、症状、检查、药物、科室等信息抽取。
- RAG / 混合检索：sentence-transformers embedding、FAISS 向量库、BM25 关键词检索和 RRF 排名融合。
- 医学知识图谱：CSV 三元组建模、NetworkX 图结构、实体查询、关系筛选。
- Graph-RAG：图谱结构化证据与文本检索证据融合生成回答。
- 医疗数据合规：使用公开数据和 demo 文本，不使用真实患者隐私数据。
- Streamlit 工程展示：项目状态、模型指标、问答证据和实体识别结果可视化。

## 当前不足

- NER 当前使用 CPU 友好子集训练，尚未完成全量 CMeEE 训练。
- 普通 BIO 标注难以完整表达 CMeEE 中的嵌套实体，正式优化建议使用 GlobalPointer、span classification 等结构。
- RAG 和 Graph-RAG 目前采用模板式生成，没有接入本地大模型或外部大模型 API。
- 医学知识图谱主展示仍使用 demo CSV 数据；CMeIE 转换管线已具备，但尚未训练关系抽取模型，也尚未进行专家审核后合并主图谱。
- PMC Open Access、Europe PMC 等医学文献来源尚未实际接入全文知识库。
- 已提供 NER 错误分析脚本；还缺少系统性的模型对比实验和检索质量评估。

## 后续优化方向

- 将 CMeEE 的嵌套实体识别改为 GlobalPointer、TPLinker 或 span classification。
- 使用 GPU 对全量 CMeEE 进行训练，并对比 `bert-base-chinese`、`hfl/chinese-roberta-wwm-ext`、`hfl/chinese-macbert-base`。
- 审核并合并 CMeIE 转换出的医学关系三元组，扩展疾病、药物、检查、症状之间的知识图谱。
- 扩展 PMC Open Access / Europe PMC 文献，构建更大规模医学文献 RAG 知识库。
- RAG 模块接入本地大语言模型或可控的医学问答模型，增加答案一致性检查。
- 增加检索质量评估集，对比纯向量检索、BM25 和混合检索在医学问题上的召回效果。
- 加入实体标准化，将“高血压”“原发性高血压”等映射到统一医学概念。
- 增加 NER 错误分析、模型对比实验、自动化测试、Dockerfile、模型训练日志可视化和实验追踪。

## 医疗安全声明

本项目仅用于 NLP 学习、课程作业、简历展示和技术验证，不提供医疗诊断、治疗方案或用药建议。任何医学问题都应咨询具备资质的医生或医疗机构。
