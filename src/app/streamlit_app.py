"""Streamlit report-ready showcase for the Chinese medical NLP project."""

from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ner.predict import MedicalNERPredictor
from src.ner.paths import resolve_ner_model_dir
from src.ner.dataset import read_bio_file
from src.ner.lexicon_postprocess import build_entity_lexicon
from src.kg.graph_rag import MedicalGraphRAG
from src.kg.kg_builder import build_candidate_relations
from src.kg.kg_loader import load_medical_kg
from src.kg.kg_query import query_entity, triples_to_records
from src.rag.qa_chain import MedicalQAChain
from src.rag.retriever import MedicalRetriever

QA_EXAMPLES = [
    "高血压患者平时需要注意什么？",
    "MedQuAD 可以用于医疗问答系统吗？",
    "What is Adult Acute Lymphoblastic Leukemia?",
    "What are the symptoms of Abdominal aortic aneurysm?",
]
QA_EXAMPLE = QA_EXAMPLES[0]
KG_EXAMPLE = "肺炎有哪些症状，需要做什么检查？"
KG_ENTITY_EXAMPLE = "肺炎"
KG_BUILD_EXAMPLE = "肺炎患者常见症状表现为咳嗽和发热，建议进行胸部CT和血常规检查，并到呼吸内科就诊。"
NER_EXAMPLE = "患者出现头痛和胸闷，医生建议进行血压检查，并考虑使用硝苯地平。"

ENTITY_COLORS = {
    "DISEASE": "#e03131",
    "SYMPTOM": "#f08c00",
    "DRUG": "#2f9e44",
    "EXAM": "#1971c2",
    "SURGERY": "#9c36b5",
    "BODY": "#0c8599",
    "TEST": "#5f3dc4",
    "DEPARTMENT": "#364fc7",
}

ENTITY_NAMES = {
    "DISEASE": "疾病",
    "SYMPTOM": "症状",
    "DRUG": "药物",
    "EXAM": "检查",
    "SURGERY": "手术",
    "BODY": "身体部位",
    "TEST": "检验指标",
    "DEPARTMENT": "科室",
}

TRAINING_LEXICON_MIN_COUNT = 5

DEMO_ENTITY_LEXICON = {
    "肺炎": "DISEASE",
    "咳痰": "SYMPTOM",
    "多饮": "SYMPTOM",
    "多尿": "SYMPTOM",
    "头晕": "SYMPTOM",
    "胸部CT": "EXAM",
    "血常规": "EXAM",
    "C反应蛋白": "TEST",
    "血压测量": "EXAM",
    "呼吸内科": "DEPARTMENT",
    "心血管内科": "DEPARTMENT",
    "内分泌科": "DEPARTMENT",
    "高血压": "DISEASE",
    "糖尿病": "DISEASE",
    "胃炎": "DISEASE",
    "冠心病": "DISEASE",
    "脑卒中": "DISEASE",
    "头痛": "SYMPTOM",
    "胸闷": "SYMPTOM",
    "胸痛": "SYMPTOM",
    "咳嗽": "SYMPTOM",
    "发热": "SYMPTOM",
    "硝苯地平": "DRUG",
    "阿莫西林": "DRUG",
    "血压检查": "EXAM",
    "胃镜检查": "EXAM",
    "心电图": "EXAM",
    "肺部CT检查": "EXAM",
    "阑尾切除术": "SURGERY",
    "内固定手术": "SURGERY",
    "右下腹": "BODY",
    "肺部": "BODY",
    "甲状腺": "BODY",
    "空腹血糖": "TEST",
    "糖化血红蛋白": "TEST",
    "肌钙蛋白": "TEST",
    "血红蛋白": "TEST",
}


@st.cache_data
def load_config() -> dict:
    """Load project configuration."""
    with (ROOT / "configs" / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path from config or status."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@st.cache_data(ttl=5)
def load_source_registry() -> list[dict[str, str]]:
    """Load public source metadata for the knowledge base."""
    registry_path = ROOT / "data" / "knowledge_base_sources.json"
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text(encoding="utf-8"))


@st.cache_data(ttl=5)
def load_project_status() -> dict[str, object]:
    """Collect lightweight project status for the dashboard."""
    configured_ner_dir = ROOT / "outputs" / "ner_model"
    ner_dir = resolve_ner_model_dir(configured_ner_dir)
    vector_dir = ROOT / "outputs" / "vector_db"
    kb_dir = ROOT / "data" / "knowledge_base"
    chunks_path = vector_dir / "chunks.json"
    backend_path = vector_dir / "backend.txt"
    eval_path = ner_dir / "eval_report.json"
    history_path = ner_dir / "training_history.json"
    vector_mtime = chunks_path.stat().st_mtime if chunks_path.exists() else 0.0

    chunks = []
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    eval_report = None
    if eval_path.exists():
        eval_report = json.loads(eval_path.read_text(encoding="utf-8"))

    training_history = []
    if history_path.exists():
        training_history = json.loads(history_path.read_text(encoding="utf-8"))

    kb_files = [
        path
        for path in kb_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv", ".json"}
    ]

    return {
        "ner_ready": (ner_dir / "model.safetensors").exists(),
        "active_model_dir": str(ner_dir),
        "vector_ready": (vector_dir / "index.faiss").exists() or (vector_dir / "embeddings.npy").exists(),
        "vector_backend": backend_path.read_text(encoding="utf-8").strip() if backend_path.exists() else "未构建",
        "kb_file_count": len(kb_files),
        "chunk_count": len(chunks),
        "vector_mtime": vector_mtime,
        "kb_sources": sorted({chunk.get("source", "") for chunk in chunks})[:8],
        "eval_report": eval_report,
        "training_history": training_history,
        "source_registry": load_source_registry(),
    }


@st.cache_resource
def load_qa_chain(config: dict, vector_mtime: float) -> MedicalQAChain:
    """Load the RAG chain once per app session."""
    _ = vector_mtime
    rag_config = config["rag"]
    qa_config = config["qa"]
    retriever = MedicalRetriever(
        vector_store_dir=str(ROOT / rag_config["vector_store_dir"]),
        embedding_model=rag_config["embedding_model"],
        top_k=int(rag_config["top_k"]),
        normalize_embeddings=bool(rag_config.get("normalize_embeddings", True)),
        retrieval_mode=str(rag_config.get("retrieval_mode", "vector")),
        bm25_top_k=_optional_int(rag_config.get("bm25_top_k")),
        rrf_rank_constant=int(rag_config.get("rrf_rank_constant", 60)),
    )
    return MedicalQAChain(
        retriever=retriever,
        disclaimer=qa_config["disclaimer"],
        max_evidence_chars=int(qa_config["max_evidence_chars"]),
    )


@st.cache_resource
def load_kg() -> object:
    """Load the lightweight CSV medical knowledge graph."""
    return load_medical_kg(ROOT / "data" / "medical_kg" / "medical_kg.csv")


@st.cache_resource
def load_graph_rag_chain(config: dict, vector_mtime: float) -> MedicalGraphRAG:
    """Load Graph-RAG and degrade gracefully if the vector store is unavailable."""
    _ = vector_mtime
    kg = load_kg()
    retriever = None
    try:
        rag_config = config["rag"]
        retriever = MedicalRetriever(
            vector_store_dir=str(ROOT / rag_config["vector_store_dir"]),
            embedding_model=rag_config["embedding_model"],
            top_k=int(rag_config["top_k"]),
            normalize_embeddings=bool(rag_config.get("normalize_embeddings", True)),
            retrieval_mode=str(rag_config.get("retrieval_mode", "vector")),
            bm25_top_k=_optional_int(rag_config.get("bm25_top_k")),
            rrf_rank_constant=int(rag_config.get("rrf_rank_constant", 60)),
        )
    except Exception:
        retriever = None
    return MedicalGraphRAG(
        kg,
        retriever=retriever,
        top_k=int(config.get("rag", {}).get("top_k", 4)),
    )


@st.cache_resource
def load_ner_predictor(
    model_dir: str,
    data_dir: str,
    train_file: str,
    train_mtime: float,
) -> MedicalNERPredictor:
    """Load the trained NER model and optional training lexicon once per app session."""
    entity_lexicon = load_training_entity_lexicon(data_dir, train_file, train_mtime)
    return MedicalNERPredictor(
        model_dir=model_dir,
        entity_lexicon=entity_lexicon,
        lexicon_min_count=TRAINING_LEXICON_MIN_COUNT,
        lexicon_add_missing=False,
    )


@st.cache_data
def load_training_entity_lexicon(
    data_dir: str,
    train_file: str,
    train_mtime: float,
) -> dict[str, object]:
    """Build a high-frequency entity lexicon from the training BIO file."""
    _ = train_mtime
    train_path = resolve_project_path(data_dir) / train_file
    if not train_path.exists():
        return {}
    tokens, labels = read_bio_file(train_path)
    return build_entity_lexicon(tokens, labels)


def get_training_lexicon_mtime(config: dict) -> float:
    """Return train BIO mtime for Streamlit cache invalidation."""
    train_path = resolve_project_path(config["ner"]["data_dir"]) / str(config["ner"]["train_file"])
    return train_path.stat().st_mtime if train_path.exists() else 0.0


def inject_css() -> None:
    """Add compact styling for a more presentation-ready Streamlit page."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .hero {
            border: 1px solid #dbe4ff;
            background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
            color: #172033;
            padding: 1.2rem 1.4rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; color: #1c4ed8; }
        .hero p { margin: 0; color: #425466; line-height: 1.55; }
        .flow {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .65rem;
            margin: 1rem 0 1.2rem 0;
        }
        .flow-step {
            border: 1px solid #d0ebff;
            background: #f8fcff;
            color: #172033;
            border-radius: 8px;
            padding: .75rem;
            min-height: 82px;
        }
        .flow-step b { display: block; margin-bottom: .25rem; color: #0b7285; }
        .flow-step span { color: #495057; font-size: .9rem; }
        .status-ok { color: #2b8a3e; font-weight: 700; }
        .status-warn { color: #c92a2a; font-weight: 700; }
        .evidence-card {
            border: 1px solid #3a3b46;
            border-radius: 8px;
            padding: .9rem 1rem;
            margin-bottom: .7rem;
            background: #262730;
            color: #f8f9fa;
        }
        .evidence-meta { color: #d0d4dc; font-size: .88rem; margin-bottom: .35rem; }
        .evidence-card code {
            color: #f8f9fa;
            background: #1b1c23;
            border-radius: 4px;
            padding: .08rem .25rem;
        }
        .evidence-card a { color: #8ecbff; }
        .answer-highlight {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: .95rem 1rem;
            margin: .65rem 0 1rem 0;
            background: #ffffff;
            color: #172033;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        .entity-tag {
            display: inline-block;
            color: white;
            border-radius: 999px;
            padding: .15rem .55rem;
            margin: .12rem;
            font-size: .85rem;
            font-weight: 600;
        }
        .note {
            border-left: 4px solid #1971c2;
            background: #f1f8ff;
            color: #172033;
            padding: .8rem 1rem;
            margin: .8rem 0;
        }
        .result-callout {
            border: 1px solid #c3fae8;
            background: #f6fffb;
            color: #173b30;
            border-radius: 8px;
            padding: .85rem 1rem;
            margin: .85rem 0;
        }
        .result-callout b { color: #087f5b; }
        @media (max-width: 900px) {
            .flow { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: dict, status: dict[str, object]) -> None:
    """Show app status and demo shortcuts in the sidebar."""
    st.sidebar.title("项目控制台")
    st.sidebar.caption("报告截图版：先看状态，再展示问答和实体识别结果。")

    st.sidebar.markdown("#### 运行状态")
    st.sidebar.markdown(_status_line("NER 模型", bool(status["ner_ready"])), unsafe_allow_html=True)
    st.sidebar.markdown(_status_line("向量库", bool(status["vector_ready"])), unsafe_allow_html=True)
    st.sidebar.write(f"向量后端：`{status['vector_backend']}`")
    st.sidebar.write(f"知识库文件：`{status['kb_file_count']}`")
    st.sidebar.write(f"检索分块：`{status['chunk_count']}`")
    st.sidebar.write(f"公开来源：`{len(status.get('source_registry', []))}`")
    try:
        kg = load_kg()
        st.sidebar.write(f"图谱实体：`{kg.entity_count()}`")
        st.sidebar.write(f"图谱关系：`{kg.relation_count()}`")
    except Exception:
        st.sidebar.write("图谱：`未就绪`")
    st.sidebar.caption(f"Active NER：{Path(str(status['active_model_dir'])).name}")

    st.sidebar.markdown("#### 快速演示")
    if st.sidebar.button("填入问答样例"):
        st.session_state["qa_question"] = QA_EXAMPLE
    if st.sidebar.button("填入 NER 样例"):
        st.session_state["ner_text"] = NER_EXAMPLE

    st.sidebar.markdown("#### 关键配置")
    st.sidebar.write(f"NER：`{config['ner']['model_name']}`")
    st.sidebar.write(f"Embedding：`{config['rag']['embedding_model']}`")
    st.sidebar.write(f"Top-K：`{config['rag']['top_k']}`")


def _status_line(label: str, ok: bool) -> str:
    class_name = "status-ok" if ok else "status-warn"
    text = "已就绪" if ok else "未就绪"
    return f"{label}：<span class='{class_name}'>{text}</span>"


def render_overview(config: dict, status: dict[str, object]) -> None:
    """Render the project overview page."""
    st.markdown(
        """
        <div class="hero">
          <h1>中文医疗文本实体识别、医学知识图谱与问答系统</h1>
          <p>面向课程报告和简历展示的医疗 NLP 项目：NER 模块抽取疾病、症状、药物、检查等实体，知识图谱模块组织结构化医学三元组，RAG/Graph-RAG 模块融合文本证据与图谱证据生成带引用的医学科普回答。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow">
          <div class="flow-step"><b>1. 合规数据</b><span>人工 demo + 可扩展公开医学数据</span></div>
          <div class="flow-step"><b>2. NER 训练</b><span>BERT Token Classification + BIO</span></div>
          <div class="flow-step"><b>3. 知识库与图谱</b><span>本地文本 + CSV 医学三元组</span></div>
          <div class="flow-step"><b>4. 证据检索</b><span>FAISS 文本检索 + 图谱查询</span></div>
          <div class="flow-step"><b>5. Graph-RAG</b><span>融合证据 + 医疗风险提示</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("核心模块", "3", "NER + KG + RAG")
    col2.metric("知识库文件", int(status["kb_file_count"]), "本地医学文本")
    col3.metric("检索分块", int(status["chunk_count"]), "可溯源证据")
    col4.metric("运行状态", "Ready", str(status["vector_backend"]).upper())

    st.markdown("### 模块状态")
    left, right = st.columns(2)
    with left:
        st.markdown("#### NER 实体识别")
        st.write("基于 Transformers 的 BERT token classification，支持 BIO 标注、训练、验证、测试和推理。页面演示采用模型输出、训练集实体词典标签纠正与医学词表补充，保证截图展示稳定。")
        eval_report = status.get("eval_report")
        if isinstance(eval_report, dict):
            with st.expander("查看 demo 评估指标", expanded=True):
                metric_cols = st.columns(3)
                metric_cols[0].metric("Precision", f"{float(eval_report['precision']):.4f}")
                metric_cols[1].metric("Recall", f"{float(eval_report['recall']):.4f}")
                metric_cols[2].metric("F1", f"{float(eval_report['f1']):.4f}")
                st.caption("demo 数据量很小，指标主要用于证明训练、保存、加载、评估流程完整；接入 CMeEE 后再做正式效果对比。")
        else:
            st.info("尚未生成评估报告。运行 `py scripts/evaluate_ner.py` 后这里会显示 Precision、Recall、F1。")

    with right:
        st.markdown("#### RAG 医疗问答")
        st.write("支持 txt、md、csv、json 本地医学文本，使用 sentence-transformers 向量化并通过 FAISS/NumPy 检索。")
        sources = status.get("kb_sources", [])
        if sources:
            st.caption("当前检索来源")
            for source in sources:
                st.code(str(source), language=None)
        else:
            st.info("尚未构建向量库。运行 `py scripts/build_vector_db.py` 后这里会显示知识来源。")

    source_registry = status.get("source_registry", [])
    if isinstance(source_registry, list) and source_registry:
        st.markdown("### 公开知识来源")
        source_df = pd.DataFrame(source_registry)
        st.dataframe(
            source_df[["name", "language", "license", "usage", "url"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 数据合规说明")
    st.markdown(
        """
        <div class="note">
        本项目使用人工编写 demo 文本，并在 README 中说明 CBLUE/CMeEE、MedQuAD、PMC Open Access 等公开来源的接入方式。
        不使用真实患者病历，不采集隐私数据，不做确定性诊断。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 最终截图建议")
    st.write("1. 先展示本页状态：模型、向量库、知识库和评估指标。")
    st.write("2. 打开 RAG 页签，输入高血压问题，展示答案、证据片段、来源和免责声明。")
    st.write("3. 打开 NER 页签，输入医学句子，展示实体表格、彩色标签和类型统计。")
    st.write("4. 打开知识图谱与 Graph-RAG 页签，输入肺炎问题，展示三元组证据、文本证据和融合回答。")


def render_qa_tab(config: dict, status: dict[str, object]) -> None:
    """Render the RAG QA experience."""
    st.subheader("医疗问答 RAG")
    st.caption("从本地医学知识库检索证据，再生成带引用编号的学习参考回答。")

    source_registry = status.get("source_registry", [])
    if isinstance(source_registry, list) and source_registry:
        with st.expander("当前知识库公开来源"):
            st.dataframe(
                pd.DataFrame(source_registry)[["name", "license", "usage", "url"]],
                use_container_width=True,
                hide_index=True,
            )

    if "qa_question" not in st.session_state:
        st.session_state["qa_question"] = QA_EXAMPLE

    st.markdown("#### 一键示例问题")
    example_cols = st.columns(2)
    for idx, example in enumerate(QA_EXAMPLES):
        with example_cols[idx % 2]:
            if st.button(example, key=f"qa_example_{idx}", use_container_width=True):
                st.session_state["qa_question"] = example
                st.rerun()

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area("输入医学问题", key="qa_question", height=110)
    with col2:
        st.metric("Top-K", int(config["rag"]["top_k"]))
        st.metric("知识库文件", int(status["kb_file_count"]))
        st.metric("检索分块", int(status["chunk_count"]))

    if st.button("生成回答", type="primary", use_container_width=True):
        try:
            chain = load_qa_chain(config, float(status.get("vector_mtime", 0.0)))
            result = chain.answer(question)
            evidence = result["evidence"]

            st.markdown("### 回答")
            render_answer_card(str(result["answer"]))

            st.markdown("### 检索证据")
            st.caption(f"共检索到 {len(evidence)} 条证据。相似度分数越高，表示与问题越相关。")
            for idx, item in enumerate(evidence, start=1):
                render_evidence_card(idx, item)
        except Exception as exc:  # noqa: BLE001 - Streamlit should show setup hints.
            st.error(str(exc))
            st.info("请先运行 `py scripts/build_vector_db.py` 构建本地向量库。")


def render_evidence_card(idx: int, item: dict[str, object]) -> None:
    """Render a cited evidence chunk."""
    source = html.escape(str(item["source"]))
    score = float(item["score"])
    content = html.escape(str(item["content"]))
    metadata = item.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    dataset = html.escape(str(metadata.get("source_dataset", "本地知识库")))
    license_text = html.escape(str(metadata.get("license", "未声明")))
    source_url = str(metadata.get("source_url", "")).strip()
    note = html.escape(str(metadata.get("note", "")))
    url_html = (
        f" · <a href='{html.escape(source_url)}' target='_blank'>来源链接</a>"
        if source_url.startswith("http")
        else ""
    )
    note_html = f"<div class='evidence-meta'>说明：{note}</div>" if note else ""
    st.markdown(
        f"""
        <div class="evidence-card">
          <div class="evidence-meta">证据 [{idx}] · 来源：<code>{source}</code> · 相似度：{score:.4f}</div>
          <div class="evidence-meta">数据集：{dataset} · 许可证：{license_text}{url_html}</div>
          {note_html}
          <div>{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_card(answer: str) -> None:
    """Render generated answer with the former evidence highlight style."""
    st.markdown(
        f"""
        <div class="answer-highlight">{html.escape(answer)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_kg_tab(config: dict, status: dict[str, object]) -> None:
    """Render the medical KG and Graph-RAG page."""
    st.subheader("医学知识图谱与 Graph-RAG")
    st.caption("使用 CSV + NetworkX 构建轻量医学知识图谱，并融合 FAISS 文本检索形成 Graph-RAG 问答流程。")

    try:
        kg = load_kg()
    except Exception as exc:  # noqa: BLE001
        st.error(f"医学知识图谱加载失败：{exc}")
        st.info("请确认 `data/medical_kg/medical_kg.csv` 存在且包含 head/relation/tail/head_type/tail_type/source 字段。")
        return

    st.markdown(
        """
        <div class="note">
        Graph-RAG 流程：用户问题 → 实体匹配 → 图谱三元组检索 → FAISS 文本证据检索 → 模板式融合回答 → 医疗风险提示。
        图谱为 demo 数据，适合展示工程链路；正式医学知识库需要专家审核和来源追踪。
        </div>
        """,
        unsafe_allow_html=True,
    )

    relation_counts = kg.relation_type_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("图谱实体", kg.entity_count())
    col2.metric("图谱关系", kg.relation_count())
    col3.metric("关系类型", len(relation_counts))

    st.markdown("### 关系类型统计")
    if relation_counts:
        st.dataframe(
            pd.DataFrame(
                [{"relation": relation, "count": count} for relation, count in relation_counts.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 实体查询")
    entity = st.text_input("输入医学实体", value=KG_ENTITY_EXAMPLE)
    relation_options = ["全部"] + list(relation_counts.keys())
    selected_relation = st.selectbox("关系类型筛选", options=relation_options)
    triples = query_entity(kg, entity, relation_type=selected_relation)
    if triples:
        st.dataframe(pd.DataFrame(triples_to_records(triples)), use_container_width=True, hide_index=True)
    else:
        st.info("没有查询到相关三元组。可以尝试输入：肺炎、糖尿病、高血压、二甲双胍、阿司匹林。")

    st.markdown("### Graph-RAG 医疗问答")
    if "graph_rag_question" not in st.session_state:
        st.session_state["graph_rag_question"] = KG_EXAMPLE
    graph_question = st.text_area("输入医学问题", key="graph_rag_question", height=90)
    if st.button("运行 Graph-RAG", type="primary", use_container_width=True):
        chain = load_graph_rag_chain(config, float(status.get("vector_mtime", 0.0)))
        result = chain.answer(graph_question)
        st.markdown("#### 识别/匹配实体")
        matched_entities = result.get("matched_entities", [])
        st.write("、".join(matched_entities) if matched_entities else "未匹配到图谱实体")

        st.markdown("#### 最终回答")
        render_answer_card(str(result["answer"]))

        st.markdown("#### 图谱结构化证据")
        graph_evidence = result.get("graph_evidence", [])
        if graph_evidence:
            st.dataframe(pd.DataFrame(graph_evidence), use_container_width=True, hide_index=True)
        else:
            st.info("未检索到图谱证据。")

        st.markdown("#### 文本检索证据")
        text_evidence = result.get("text_evidence", [])
        if text_evidence:
            for idx, item in enumerate(text_evidence, start=1):
                render_evidence_card(idx, item)
        else:
            st.info("未检索到文本证据，或向量库尚未构建。")

    st.markdown("### NER 辅助图谱构建")
    st.caption("候选关系仅用于展示和人工审核，默认不写入主图谱，避免污染 demo 数据。")
    if "kg_build_text" not in st.session_state:
        st.session_state["kg_build_text"] = KG_BUILD_EXAMPLE
    build_text = st.text_area("输入医学文本", key="kg_build_text", height=100)
    if st.button("抽取候选关系", use_container_width=True):
        try:
            predictor = load_ner_predictor(
                str(status["active_model_dir"]),
                str(config["ner"]["data_dir"]),
                str(config["ner"]["train_file"]),
                get_training_lexicon_mtime(config),
            )
            model_entities = predictor.predict(build_text, max_length=int(config["ner"]["max_length"]))
            lexicon_entities = demo_lexicon_entities(build_text)
            entities = merge_entities(model_entities, lexicon_entities)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"NER 模型暂不可用，已使用页面词表补充进行演示：{exc}")
            entities = demo_lexicon_entities(build_text)

        st.markdown("#### NER 实体")
        if entities:
            entity_df = pd.DataFrame(entities)
            entity_df["type_name"] = entity_df["type"].map(ENTITY_NAMES).fillna(entity_df["type"])
            st.dataframe(entity_df[["text", "type", "type_name", "start", "end"]], use_container_width=True, hide_index=True)
            render_entity_tags(entities)
        else:
            st.info("没有识别到实体。")

        candidates = build_candidate_relations(build_text, entities)
        st.markdown("#### 候选关系")
        if candidates:
            st.dataframe(
                pd.DataFrame([candidate.to_dict() for candidate in candidates]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("没有生成候选关系。可尝试包含疾病、症状/检查/科室以及触发词，例如“表现为、建议、检查、就诊”。")


def render_ner_tab(config: dict, status: dict[str, object]) -> None:
    """Render the NER demo experience."""
    st.subheader("中文医疗实体识别")
    st.caption("识别疾病、症状、药物、检查、手术、身体部位和检验指标。报告演示采用模型输出 + 训练集实体词典标签纠正 + 医学词表补充，保证结果稳定可展示。")

    if "ner_text" not in st.session_state:
        st.session_state["ner_text"] = NER_EXAMPLE

    text = st.text_area("输入中文医疗文本", key="ner_text", height=120)
    if st.button("识别实体", type="primary", use_container_width=True):
        try:
            predictor = load_ner_predictor(
                str(status["active_model_dir"]),
                str(config["ner"]["data_dir"]),
                str(config["ner"]["train_file"]),
                get_training_lexicon_mtime(config),
            )
            model_entities = predictor.predict(text, max_length=int(config["ner"]["max_length"]))
            lexicon_entities = demo_lexicon_entities(text)
            entities = merge_entities(model_entities, lexicon_entities)
            if not entities:
                st.info("没有识别到实体。可以换用 README 中的演示句子，或接入更大的 CMeEE 数据重新训练模型。")
                return

            st.markdown(
                """
                <div class="result-callout">
                当前展示为 <b>BERT-NER 模型输出 + 训练集实体词典标签纠正 + 医学词表补充</b> 的演示结果，适合报告截图展示；正式实验评价请以评估脚本输出为准。
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_entity_summary(entities)
            render_entity_tags(entities)

            df = pd.DataFrame(entities)
            df["type_name"] = df["type"].map(ENTITY_NAMES).fillna(df["type"])
            st.markdown("### 实体明细")
            st.dataframe(
                df[["text", "type", "type_name", "start", "end"]],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.info("请先运行 `py scripts/train_ner.py` 训练并保存 NER 模型。")


def demo_lexicon_entities(text: str) -> list[dict[str, object]]:
    """Exact-match fallback for live demos when the tiny model predicts nothing."""
    entities: list[dict[str, object]] = []
    occupied: set[int] = set()
    for term, entity_type in sorted(DEMO_ENTITY_LEXICON.items(), key=lambda item: len(item[0]), reverse=True):
        start = text.find(term)
        while start != -1:
            end = start + len(term)
            if not any(idx in occupied for idx in range(start, end)):
                entities.append({"text": term, "type": entity_type, "start": start, "end": end})
                occupied.update(range(start, end))
            start = text.find(term, end)
    return sorted(entities, key=lambda item: int(item["start"]))


def merge_entities(
    model_entities: list[dict[str, object]],
    lexicon_entities: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge model entities with lexicon entities while avoiding overlaps."""
    merged = sorted(model_entities, key=lambda item: (int(item["start"]), int(item["end"])))
    occupied = {
        idx
        for entity in merged
        for idx in range(int(entity["start"]), int(entity["end"]))
    }

    for entity in lexicon_entities:
        start = int(entity["start"])
        end = int(entity["end"])
        if any(idx in occupied for idx in range(start, end)):
            continue
        merged.append(entity)
        occupied.update(range(start, end))

    return sorted(merged, key=lambda item: int(item["start"]))


def render_entity_summary(entities: list[dict[str, object]]) -> None:
    """Show entity type counts."""
    counts = Counter(str(entity["type"]) for entity in entities)
    st.markdown("### 实体类型统计")
    cols = st.columns(min(4, max(1, len(counts))))
    for idx, (entity_type, count) in enumerate(counts.items()):
        label = ENTITY_NAMES.get(entity_type, entity_type)
        cols[idx % len(cols)].metric(label, count)


def render_entity_tags(entities: list[dict[str, object]]) -> None:
    """Show colored entity pills."""
    tags = []
    for entity in entities:
        entity_type = str(entity["type"])
        color = ENTITY_COLORS.get(entity_type, "#495057")
        text = html.escape(str(entity["text"]))
        label = ENTITY_NAMES.get(entity_type, entity_type)
        tags.append(
            f"<span class='entity-tag' style='background:{color}'>{text} · {label}</span>"
        )
    st.markdown("### 彩色实体标签")
    st.markdown("".join(tags), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="中文医疗 NLP + KG + RAG 项目展示", layout="wide")
    inject_css()
    config = load_config()
    status = load_project_status()
    render_sidebar(config, status)

    overview_tab, qa_tab, ner_tab, kg_tab = st.tabs(
        ["项目总览", "医疗问答 RAG", "医疗 NER", "医学知识图谱与 Graph-RAG"]
    )
    with overview_tab:
        render_overview(config, status)
    with qa_tab:
        render_qa_tab(config, status)
    with ner_tab:
        render_ner_tab(config, status)
    with kg_tab:
        render_kg_tab(config, status)


if __name__ == "__main__":
    main()
