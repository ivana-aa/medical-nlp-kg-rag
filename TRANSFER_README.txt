医疗 NLP 项目迁移包
生成时间：2026-06-11 20:26:44

解压后目录：medical_nlp_project

已包含：
- 源码、配置、脚本、测试、README、阶段报告
- RAG 知识库 data/knowledge_base
- 医学知识图谱 data/medical_kg，包括 CMeIE 派生 CSV
- NER processed 数据 data/processed/ner_train，用于推理词典和继续评估
- Active NER 模型 outputs/ner_model_runs/20260603_211731
- Active NER 指针 outputs/ner_model_active.txt
- RAG 向量库 outputs/vector_db
- logs/reports outputs/logs

未包含：
- .venv、.pip-cache、.hf-cache、.tmp、__pycache__ 等本机缓存
- data/raw 原始下载数据，可按 README 重新下载
- models/chinese-macbert-base 本地基座模型；active checkpoint 已包含展示推理所需权重和 tokenizer
- 历史 NER 训练 run，只保留当前 active 模型

原电脑恢复步骤：
1. 解压到目标目录，例如 D:\medical_nlp_project\medical_nlp_project
2. 进入目录：cd D:\medical_nlp_project\medical_nlp_project
3. 创建虚拟环境：python -m venv .venv
4. 激活环境：.\.venv\Scripts\activate
5. 安装依赖：pip install -r requirements.txt
6. 验证测试：python -m unittest discover -s tests
7. 启动页面：streamlit run src\app\streamlit_app.py

关键检查：
- outputs\ner_model_active.txt 存在
- outputs\ner_model_runs\20260603_211731 存在
- outputs\vector_db\chunks.json 存在
- data\medical_kg\medical_kg.csv 存在
