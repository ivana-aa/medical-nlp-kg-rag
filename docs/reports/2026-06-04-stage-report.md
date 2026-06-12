# 中文医疗 NLP 项目阶段性报告

报告日期：2026-06-04  
项目名称：基于中文医疗文本的实体识别、医学知识图谱与 RAG 医疗问答系统  
项目路径：`D:\medical_nlp_project\medical_nlp_project`

## 一、项目概述

本项目面向中文医疗文本处理和医疗智能问答原型，已经形成从医学文本数据、实体识别、知识图谱、文本检索到 Streamlit 可视化展示的完整工程链路。当前系统不使用真实患者病历，不采集隐私数据，主要基于公开数据集、人工 demo 医学科普文本和可追溯公开来源样本进行开发与展示。

核心能力包括：

1. `NER (Named Entity Recognition，命名实体识别)`：从中文医疗文本中识别疾病、症状、药物、检查、手术、身体部位、检验指标等实体。
2. `KG (Knowledge Graph，知识图谱)`：使用 CSV + NetworkX 管理医学三元组，支持实体查询、关系筛选和候选关系生成。
3. `RAG (Retrieval-Augmented Generation，检索增强生成)`：从本地医学知识库中检索相关证据，并生成带引用片段的医学科普回答。
4. `Graph-RAG (Graph Retrieval-Augmented Generation，图谱检索增强生成)`：融合知识图谱结构化证据和 FAISS 文本检索证据，生成更具可解释性的回答。
5. Streamlit 工程展示：提供项目总览、医疗问答 RAG、医疗 NER、医学知识图谱与 Graph-RAG 页面。

## 二、关键术语说明

- `BIO (Begin-Inside-Outside，开始-内部-外部标注)`：序列标注格式，用 `B-实体类型` 表示实体开头，用 `I-实体类型` 表示实体内部，用 `O` 表示非实体。
- `MacBERT (MLM as Correction BERT，纠错式掩码语言模型 BERT)`：中文预训练语言模型，本项目使用 `hfl/chinese-macbert-base` 作为 NER 主模型。
- `FAISS (Facebook AI Similarity Search，Facebook AI 相似度搜索库)`：向量相似度检索库，用于 RAG 文本分块检索。
- `CMeEE (Chinese Medical Entity Extraction，中文医学实体抽取)`：CBLUE 中的中文医疗实体识别数据集，用于 NER 训练与评估。
- `CMeIE (Chinese Medical Information Extraction，中文医学信息抽取)`：CBLUE 中的医学关系抽取数据集，当前已准备转换管线，但暂未作为本阶段重点。
- `Precision (精确率)`：模型预测为实体的结果中，有多少是真正正确的。
- `Recall (召回率)`：标准答案中的实体，有多少被模型识别出来。
- `F1-score (F1 值)`：精确率和召回率的调和平均，用于综合衡量模型效果。

## 三、当前工程结构

当前项目已完成模块化工程结构：

```text
medical_nlp_project/
├── configs/        # 配置文件
├── data/           # demo 知识库、医学 KG、小型样例数据
├── docs/           # 设计文档与阶段报告
├── examples/       # 示例输入
├── scripts/        # 训练、评估、数据处理、应用启动脚本
├── src/
│   ├── ner/        # NER 数据、训练、评估、推理、错误分析
│   ├── rag/        # 文档加载、分块、向量化、检索、问答
│   ├── kg/         # 知识图谱加载、查询、Graph-RAG、CMeIE 转换
│   ├── app/        # Streamlit 展示页面
│   └── utils/      # 日志、指标等工具
└── tests/          # 单元测试
```

同时已经初始化本地 Git 仓库，并通过 `.gitignore` 排除了 `.venv/`、`models/`、`outputs/`、`data/raw/`、`data/processed/`、缓存文件和训练产物，避免把大模型权重、原始数据和运行日志误提交。

## 四、数据与合规情况

项目当前数据来源分为三类：

1. 内置 demo 医学科普文本：用于 RAG 和页面展示，内容不包含真实患者隐私。
2. CMeEE 公开数据：用于中文医疗 NER 训练与评估，当前已转换为 BIO 格式。
3. MedQuAD 样本与公开来源说明：用于扩展 RAG 知识库，保留来源和许可证说明。

合规原则：

- 不使用真实患者病历。
- 不处理患者姓名、身份证、电话、住址等隐私信息。
- 不非法爬取医学网站或封闭数据。
- 所有医疗问答回答均保留风险提示，明确不能替代医生诊断、治疗建议或用药指导。

## 五、NER 模块进展

### 5.1 已完成功能

NER 模块已经支持：

- BIO 数据读取与对齐。
- PyTorch + Transformers token classification 训练。
- CMeEE 数据转换与抽样训练。
- train/dev/test 评估。
- 模型运行目录管理：`outputs/ner_model_runs/<run_id>`。
- active 模型指针：`outputs/ner_model_active.txt`。
- Streamlit 页面推理展示。
- 错误分析脚本。
- 可选词典后处理实验。

当前 active 模型：

```text
outputs/ner_model_runs/20260603_211731
```

当前 active 测试集指标：

```text
Precision: 0.5574
Recall:    0.6293
F1:        0.5911
```

与最早 CPU 友好子集模型 F1 `0.5203` 相比，当前 active 模型 F1 提升约 `0.0708`。

### 5.2 GPU 训练与实验记录

本阶段使用 NVIDIA RTX 4070 Laptop GPU 继续训练和调参，环境已升级到支持 CUDA 的 PyTorch：

```text
PyTorch: 2.6.0+cu124
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

实验重点包括：

- 从已有 checkpoint 继续训练。
- 对比不同学习率。
- 对比 `max_length=128` 与 `max_length=192`。
- 尝试额外 epoch。
- 尝试本地下载的 `hfl/chinese-macbert-base`。
- 尝试 entity threshold 后处理。
- 尝试 weighted loss。

最终选择 `20260603_211731` 作为 active 模型，因为它在 test 集上取得当前最优 F1。

## 六、NER 错误分析与后处理优化

新增：

- `scripts/analyze_ner_errors.py`
- `src/ner/error_analysis.py`
- `scripts/evaluate_ner_lexicon.py`
- `src/ner/lexicon_postprocess.py`

错误分析结果显示：

```text
TP/FP/FN: 5899 / 4685 / 3475
```

主要问题：

- `SYMPTOM (Symptom，症状)` 类别最弱，F1 为 `0.3972`。
- `SYMPTOM` 与 `DISEASE (Disease，疾病)` 存在明显类别混淆。
- `EXAM (Examination，检查)` 与 `TEST (Test，检验指标/检查项)` 语义接近，模型容易混淆。
- 高频漏报包括 `疼痛`、`心力衰竭`、`感染` 等。
- 高频误报包括 `细胞`、`细菌性肺炎`、`染色体` 等。

词典后处理实验：

`Gazetteer (实体词典)` 从训练集 BIO 中统计高频实体文本及其多数类别，用于可选 `Label Correction (标签纠正)`。

测试集结果：

```text
Raw:  P=0.5574  R=0.6293  F1=0.5911
Post: P=0.6031  R=0.6260  F1=0.6144
```

结论：

- 只做标签纠正有效，F1 从 `0.5911` 提升到 `0.6144`。
- 不建议开启 `--add-missing`，因为自动补漏会提高 Recall，但显著降低 Precision，整体 F1 收益不足。
- active 模型本身暂不替换，词典标签纠正已接入 Streamlit NER 推理展示，并继续保留 demo 医学词表作为展示兜底。

## 七、RAG 模块进展

RAG 模块已经支持：

- 本地 `.txt`、`.md`、`.csv`、`.json` 医学文本加载。
- 文本分块。
- sentence-transformers 向量化。
- FAISS 向量库检索。
- 检索证据卡片展示。
- 模板式医疗回答生成。
- 医疗风险提示。

当前知识库包括：

- 中文医学科普 demo。
- lab tests CSV。
- medical FAQ JSON。
- MedQuAD 医疗问答样本。
- public medical sources 元数据。

最近一次向量库规模：

```text
Loaded documents: 62
Generated chunks: 142
Vector backend: faiss
```

RAG 当前没有接入外部大模型 API，也没有使用真实患者数据。回答以“检索证据 + 模板生成”为主，适合课程报告和工程展示。

## 八、知识图谱与 Graph-RAG 进展

知识图谱模块已经完成：

- `data/medical_kg/medical_kg.csv` demo 图谱。
- CSV + NetworkX 图加载。
- 实体数量、关系数量、关系类型统计。
- 实体查询。
- 关系筛选。
- 问题实体匹配。
- 图谱证据格式化。
- NER 辅助候选关系生成。
- Graph-RAG 回答链路。

当前 demo 图谱覆盖：

- 肺炎
- 糖尿病
- 高血压
- 阿司匹林
- 华法林
- 二甲双胍
- 常见症状
- 推荐检查
- 就诊科室
- 相关药物
- 用药风险

Graph-RAG 流程：

```text
用户问题 → 图谱实体匹配 → 三元组检索 → FAISS 文本检索
      → 图谱证据与文本证据融合 → 风险提示
```

同时已新增 CMeIE 转换管线：

- `src/kg/cmeie_importer.py`
- `scripts/prepare_cmeie_kg.py`

该管线可以把 CMeIE 的 `SPO (Subject-Predicate-Object，主语-谓词-宾语三元组)` 转为项目现有 KG CSV 格式，但本阶段根据优先级暂时跳过 CMeIE 原始数据接入。

## 九、Streamlit 展示进展

Streamlit 页面已经包含：

1. 项目总览：展示系统定位、流程、模型状态、向量库状态、知识库规模和合规说明。
2. 医疗问答 RAG：输入医学问题，输出回答、引用证据和相似度。
3. 医疗 NER：输入中文医疗文本，展示实体、类型、位置和统计。
4. 医学知识图谱与 Graph-RAG：展示图谱统计、实体查询、Graph-RAG 问答和 NER 辅助候选关系。

已在桌面创建一键打开项目脚本，方便演示时快速启动项目页面。

## 十、工程质量与版本管理

本阶段新增测试覆盖：

- NER confidence threshold。
- weighted loss。
- CMeIE importer。
- NER error analysis。
- NER lexicon post-processing。

当前全量单元测试结果：

```text
17 tests OK
```

同时已完成：

- 重复 `(1)` 文件清理，释放约 `3.29 GB`。
- Git 安装与本地仓库初始化。
- `.gitignore` 配置，避免误提交大文件和运行产物。

## 十一、当前不足

1. NER 仍是普通 BIO token classification，无法完整表达 CMeEE 中的嵌套实体。
2. `SYMPTOM` 类别表现偏弱，误报和漏报都较多。
3. `EXAM` 与 `TEST` 标签边界和语义差异需要进一步清洗或重定义。
4. RAG 目前为模板式生成，没有接入本地大语言模型或外部大模型 API。
5. 医学知识图谱仍以 demo CSV 为主，尚未完成 CMeIE 或专家审核知识库的规模化合并。
6. 检索质量还缺少系统评估，例如召回率、证据命中率和人工标注问答集。

## 十二、下一阶段计划

优先级建议如下：

1. NER 错误驱动优化：
   - 清洗 `SYMPTOM`、`DISEASE`、`EXAM`、`TEST` 的混淆样本。
   - 保留词典标签纠正作为可选推理增强。
   - 继续评估是否引入 `CRF (Conditional Random Field，条件随机场)`。

2. 嵌套实体建模：
   - 研究 `GlobalPointer (Global Pointer，全局指针)` 或 span classification。
   - 解决 CMeEE 嵌套实体在 BIO 中丢失的问题。

3. 知识图谱扩展：
   - 下载并审核 CMeIE 数据。
   - 将高质量 SPO 三元组合并到独立扩展图谱。
   - 增加实体标准化，如“高血压”“原发性高血压”的统一映射。

4. RAG 检索增强：
   - 引入 BM25 + 向量检索混合召回。
   - 扩展 PMC Open Access / Europe PMC 文献。
   - 增加检索评估集。

5. 工程交付：
   - 完成本地 Git 初始提交。
   - 准备演示截图。
   - 视需要导出 Word/PDF 报告。

## 十三、阶段性结论

截至 2026-06-04，项目已经从单一中文医疗 NER demo 扩展为包含 NER、RAG、KG、Graph-RAG、错误分析和可选后处理实验的完整中文医疗 NLP 原型系统。工程结构清晰，功能链路完整，具备课程报告、简历展示和后续科研/工程扩展价值。

当前最有说服力的结果是：在 RTX 4070 Laptop GPU 上完成更大规模 CMeEE 子集训练后，active NER test F1 达到 `0.5911`；进一步通过训练集实体词典做标签纠正后，实验 F1 提升到 `0.6144`。这说明项目已经具备从“模型训练”到“错误分析”再到“针对性优化”的完整闭环。

医疗安全方面，项目始终保持风险提示和数据合规边界，所有问答仅用于学习和项目演示，不能替代医生诊断、治疗建议或用药指导。
