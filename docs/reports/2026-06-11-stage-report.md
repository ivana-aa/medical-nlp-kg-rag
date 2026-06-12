# 项目阶段性进度报告（2026-06-11）

项目名称：基于中文医疗文本的实体识别、医学知识图谱与 RAG 医疗问答系统  
项目路径：`D:\medical_nlp_project\medical_nlp_project`  
报告日期：2026-06-11  

## 一、当前结论

截至 2026-06-11，项目已经从基础中文医疗 NER demo 扩展为一个包含 `NER (Named Entity Recognition，命名实体识别)`、`KG (Knowledge Graph，知识图谱)`、`RAG (Retrieval-Augmented Generation，检索增强生成)`、`Graph-RAG (Graph Retrieval-Augmented Generation，图谱检索增强生成)`、CMeIE 关系数据接入、混合检索和 Streamlit 展示优化的完整中文医疗 NLP 原型系统。

当前系统重点不是“单个模型指标最高”，而是形成了完整工程链路：

```text
公开/演示医学文本
→ CMeEE/CMeIE 数据处理
→ MacBERT 医疗 NER
→ 医学知识图谱
→ FAISS 向量检索 + BM25 关键词检索
→ RRF 混合检索融合
→ RAG / Graph-RAG 问答
→ Streamlit 展示与医疗风险提示
```

项目仍保持安全边界：不使用真实患者病历，不采集隐私数据，所有医疗问答均保留风险提示，不能替代医生诊断、治疗建议或用药指导。

## 二、核心术语说明

1. `NER (Named Entity Recognition，命名实体识别)`：从文本中识别疾病、症状、药物、检查、身体部位、检验指标等实体。
2. `BIO (Begin-Inside-Outside，开始-内部-外部标注)`：序列标注格式，用 B/I/O 标记实体边界，适合普通实体识别，但难以完整表达嵌套实体。
3. `MacBERT (MLM as Correction BERT，纠错式掩码语言模型 BERT)`：中文预训练语言模型，本项目使用 `hfl/chinese-macbert-base` 做 token classification。
4. `RAG (Retrieval-Augmented Generation，检索增强生成)`：先检索证据，再生成回答，优点是答案可追溯。
5. `FAISS (Facebook AI Similarity Search，Facebook AI 相似度搜索库)`：向量相似度检索库，用于本地知识库分块检索。
6. `BM25 (Best Matching 25，经典关键词检索排序算法)`：基于关键词出现频率和文档长度的检索算法，对医学词、药名、检查名等精确匹配更稳定。
7. `RRF (Reciprocal Rank Fusion，倒数排名融合)`：融合多个检索排行榜的方法，本项目用于合并向量检索和 BM25 检索结果。
8. `KG (Knowledge Graph，知识图谱)`：用实体和关系组织结构化医学知识，例如 `肺炎 - 常见症状 - 发热`。
9. `Graph-RAG (Graph Retrieval-Augmented Generation，图谱检索增强生成)`：融合知识图谱三元组证据和文本检索证据生成回答。
10. `CMeIE (Chinese Medical Information Extraction，中文医学信息抽取)`：CBLUE 中的医学关系抽取数据集，提供 `SPO (Subject-Predicate-Object，主语-谓词-宾语三元组)` 标注。

## 三、当前工程结构

主要目录：

```text
configs/              # 项目配置
data/knowledge_base/  # RAG 本地医学知识库
data/medical_kg/      # demo KG 和 CMeIE 派生 KG
docs/reports/         # 阶段报告
scripts/              # 数据准备、训练、评估、图谱构建脚本
src/app/              # Streamlit 展示页面
src/kg/               # 知识图谱、CMeIE 转换与过滤、Graph-RAG
src/ner/              # NER 数据集、模型、推理、后处理、错误分析
src/rag/              # 文档加载、切分、向量库、BM25、混合检索、QA chain
tests/                # 自动化测试
```

当前关键脚本：

```text
scripts/train_ner.py
scripts/evaluate_ner.py
scripts/evaluate_ner_lexicon.py
scripts/analyze_ner_errors.py
scripts/build_vector_db.py
scripts/download_cmeie_from_hf.py
scripts/prepare_cmeie_kg.py
scripts/build_cmeie_review_kg.py
scripts/run_app.py
```

## 四、NER 模块进展

当前 NER 模块已经具备：

- 使用 PyTorch + Transformers。
- 模型基础为 `hfl/chinese-macbert-base`。
- 支持 CMeEE / CBLUE 风格数据转换。
- 支持 BIO 标注、训练、验证、测试、评估和推理。
- 支持 CUDA GPU 训练。
- 支持错误分析、实体置信度阈值实验、类别权重实验和词典后处理实验。

当前 active NER 模型：

```text
outputs/ner_model_runs/20260603_211731
```

当前 active NER 测试集指标：

```text
Precision: 0.5574
Recall:    0.6293
F1:        0.5911
```

训练历史记录：

```text
epoch: 1
loss: 0.0508
dev_f1: 0.5970
```

与早期 CPU 友好子集模型 F1 0.5203 相比，当前 active 模型 F1 提升约 0.0708。  
后续词典标签纠正实验进一步提升了测试效果：

```text
Raw P/R/F1:  0.5574 / 0.6293 / 0.5911
Post P/R/F1: 0.6031 / 0.6260 / 0.6144
```

当前决策：

- 保持 `20260603_211731` 为 active checkpoint。
- 在 Streamlit 推理端使用高频训练实体词典进行标签纠正。
- 不启用 add-missing 词典补全，因为它会明显降低 Precision。

## 五、RAG 模块进展

当前 RAG 模块已经具备：

- 支持 `.txt`、`.md`、`.csv`、`.json` 本地医学文本。
- 使用 `sentence-transformers` 生成向量。
- 使用 FAISS 作为向量检索后端。
- 支持 Windows 下 FAISS 不可用时的 NumPy fallback。
- 回答包含引用证据、来源、许可证信息和医疗风险提示。

当前向量库状态：

```text
vector_store_dir: outputs/vector_db
retrieval chunks: 142
backend: faiss
embedding model: shibing624/text2vec-base-chinese
top_k: 4
```

已新增混合检索：

```text
retrieval_mode: hybrid
bm25_top_k: 8
rrf_rank_constant: 60
```

技术含义：

- `Vector Retrieval (向量检索)`：解决语义近似问题。
- `BM25 (Best Matching 25，关键词检索)`：增强精确医学词匹配能力。
- `RRF (Reciprocal Rank Fusion，倒数排名融合)`：稳定融合两种排序结果。

当前价值：

- 对“二甲双胍有哪些用药风险？”这类带明确药名的问题，检索更稳定。
- 不改变 RAG chain 的返回结构，因此兼容原有 Streamlit 和 Graph-RAG。

## 六、医学知识图谱与 Graph-RAG 进展

主图谱仍使用 demo CSV：

```text
data/medical_kg/medical_kg.csv
```

当前主图谱规模：

```text
entities: 24
relations: 20
```

主图谱覆盖：

- 肺炎、糖尿病、高血压等疾病。
- 常见症状、推荐检查、就诊科室、相关药物。
- 阿司匹林、华法林、二甲双胍等用药风险。

Graph-RAG 当前能力：

```text
用户问题
→ 图谱实体匹配
→ 图谱三元组检索
→ FAISS/BM25/RRF 文本证据检索
→ 图谱证据 + 文本证据融合回答
→ 医疗风险提示
```

NER 辅助图谱构建能力已经加入：

- 输入医学文本。
- 调用 NER 抽取实体。
- 用规则生成候选关系。
- 候选关系仅展示，不默认写入主图谱，避免污染 demo 数据。

## 七、CMeIE 接入进展

已从 Hugging Face 数据集仓库接入：

```text
https://huggingface.co/datasets/Aunderline/CMeIE
```

新增脚本：

```text
scripts/download_cmeie_from_hf.py
scripts/prepare_cmeie_kg.py
scripts/build_cmeie_review_kg.py
```

下载并转换结果：

```text
records:         22459
spo_items:       54286
valid_triples:   54286
written_triples: 41347
```

生成的独立 CMeIE 图谱：

```text
data/medical_kg/medical_kg_cmeie.csv
```

该文件不会覆盖主图谱。

## 八、CMeIE 审核版子图谱

为避免直接把 4 万多条关系合并到展示图谱中，项目已增加审核版子图谱过滤流程。

新增模块：

```text
src/kg/cmeie_filter.py
```

新增脚本：

```text
scripts/build_cmeie_review_kg.py
```

过滤策略：

- `Schema Mapping (模式映射)`：将 CMeIE 原始关系归并到项目展示更容易理解的关系类型。
- `Quality Filter (质量过滤)`：过滤空实体、过长实体、Markdown/URL 噪声。
- `Deduplication (去重)`：过滤重复三元组。
- `Limit Control (数量上限控制)`：限制每类关系和每个头实体的数量，避免高频关系压倒展示。

真实数据过滤结果：

```text
input_triples:              41346
kept_triples:               13715
dropped_unmapped_relation:  11161
dropped_invalid_text:       216
dropped_duplicates:         53
dropped_by_limits:          16201
```

审核版子图谱：

```text
data/medical_kg/medical_kg_cmeie_reviewed.csv
```

审核版图谱规模：

```text
entities: 11206
relations: 13715
```

主要关系分布：

```text
常见症状: 2000
推荐检查: 2000
病因:     2000
相关药物: 2000
同义词:   1868
并发症:   1446
风险因素: 1094
发病部位: 948
预防:     239
发病机制: 54
传播途径: 45
就诊科室: 21
```

当前决策：

- CMeIE 审核版子图谱已经生成并可加载。
- 暂不自动合并到主图谱。
- 后续建议先做抽样审核和关系白名单，再决定是否加入 Graph-RAG 展示入口。

## 九、Streamlit 展示页进展

当前 Streamlit 页面已经包含：

- 项目总览。
- 医疗问答 RAG。
- 医疗 NER。
- 医学知识图谱与 Graph-RAG。
- NER 辅助候选关系展示。

近期展示层优化：

- RAG 检索证据卡片改为侧栏同款深灰底、白色字体，降低视觉抢占。
- 回答区域改为原证据卡片的浅色高亮样式，突出最终回答。
- Graph-RAG 最终回答同步使用高亮回答卡片。
- 桌面启动脚本已调整为不再打开项目文件夹，只打开一次 `http://localhost:8501/`。

桌面启动脚本：

```text
C:\Users\23169\Desktop\打开医疗NLP项目.cmd
```

## 十、自动化测试与验证

当前测试覆盖：

- CMeIE importer。
- CMeIE Hugging Face 下载逻辑。
- CMeIE 审核版过滤逻辑。
- NER 错误分析。
- NER 词典后处理。
- NER predictor 后处理。
- NER 置信度阈值与 loss weight。
- RAG BM25 / Hybrid Retrieval。

最新验证结果：

```text
python -m unittest discover -s tests
Ran 32 tests in 0.023s
OK
```

其他验证：

- active NER 模型指针可读取。
- FAISS 向量库可读取。
- 主图谱 `medical_kg.csv` 可加载。
- CMeIE 审核版子图谱 `medical_kg_cmeie_reviewed.csv` 可加载。
- Streamlit 样式调整已通过 Python 语法检查。

## 十一、本地 Git 版本记录

近期关键提交：

```text
f9b5ea4 Adjust RAG answer and evidence contrast
c60a890 Add reviewed CMeIE KG filtering
2a126e5 Add Hugging Face CMeIE download pipeline
033a94d Add hybrid retrieval for medical RAG
5689e1d Add NER lexicon correction to inference demo
e49c358 Initial medical NLP project baseline
```

当前代码层面已经把重要阶段保存为本地 Git commit。  
大数据和生成文件仍通过 `.gitignore` 管理，例如：

```text
data/raw/
data/processed/
data/medical_kg/medical_kg_cmeie.csv
data/medical_kg/medical_kg_cmeie_reviewed.csv
models/
outputs/
```

## 十二、当前不足

1. NER 仍是普通 BIO token classification，无法完整解决 CMeEE 嵌套实体问题。
2. active NER 模型 F1 约 0.5911，词典后处理后约 0.6144，仍有较大优化空间。
3. RAG 当前仍是模板式生成，没有接入本地大语言模型或外部大模型 API。
4. CMeIE 审核版子图谱尚未人工抽样审核，暂不适合直接作为正式医学知识库。
5. Graph-RAG 当前主展示仍使用小型 demo 图谱，尚未接入大规模审核版 CMeIE 子图谱。
6. 尚未建立系统化 RAG 检索评估集，无法量化比较 vector、BM25、hybrid 三种检索模式。
7. 桌面启动脚本是本机辅助文件，不属于 Git 跟踪文件。

## 十三、下一步建议

短期优先级：

1. 建立 CMeIE 审核抽样表：从审核版子图谱中按关系类型抽样，人工检查是否适合展示。
2. 给 Graph-RAG 增加可选图谱来源：demo 主图谱 / CMeIE 审核版子图谱二选一，但默认仍使用 demo 主图谱。
3. 构建 RAG 检索评估集：准备 20 到 50 个问题，标注期望命中的知识片段，对比 vector、BM25、hybrid。
4. 为 Streamlit 增加“当前检索模式”显示，展示 `hybrid`、`bm25_top_k` 和 `rrf_rank_constant`。

中期优化方向：

1. NER 方向：研究 `GlobalPointer (全局指针)`、`Span Classification (片段分类)` 或 `CRF (Conditional Random Field，条件随机场)`。
2. KG 方向：完成 CMeIE 关系白名单、实体标准化和图谱合并策略。
3. RAG 方向：接入更可靠的重排序模型或本地 LLM，同时保持医疗风险提示。
4. 工程方向：增加实验记录、Dockerfile、自动启动脚本版本化和展示用一键检查脚本。

## 十四、展示建议

推荐演示顺序：

1. 项目总览：说明这是一个整合 NER、KG、RAG、Graph-RAG 的中文医疗 NLP 工程项目。
2. 医疗问答 RAG：输入“高血压患者平时需要注意什么？”展示混合检索证据和风险提示。
3. 医疗 NER：输入“患者出现头痛和胸闷，医生建议进行血压检查，并考虑使用硝苯地平。”展示实体识别结果。
4. 医学知识图谱与 Graph-RAG：输入“肺炎有哪些症状，需要做什么检查？”展示图谱证据、文本证据和最终回答。
5. CMeIE 数据接入说明：展示已从公开 Hugging Face 数据集下载、转换和过滤，但不会直接污染主图谱。

## 十五、总结

项目当前已经具备较完整的医疗 NLP 原型系统能力：

- NER 训练与推理链路完整。
- RAG 检索和回答链路完整。
- KG 与 Graph-RAG 链路完整。
- CMeIE 关系数据已经完成下载、转换、过滤和审核版子图谱生成。
- Streamlit 展示界面已经针对报告演示做过多轮优化。
- 自动化测试目前 32 项通过。

下一阶段的重点不应盲目继续堆功能，而应围绕“可解释、可验证、可展示”推进：先做 CMeIE 抽样审核和 RAG 检索评估，再考虑更复杂的关系抽取模型训练或嵌套实体模型改造。
