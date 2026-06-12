import unittest

from src.rag.bm25 import BM25TextIndex, reciprocal_rank_fusion
from src.rag.retriever import MedicalRetriever
from src.rag.splitter import TextChunk


def make_chunk(chunk_id: str, content: str) -> TextChunk:
    return TextChunk(
        content=content,
        source="unit_test",
        chunk_id=chunk_id,
        metadata={},
    )


class BM25TextIndexTests(unittest.TestCase):
    def test_bm25_prioritizes_exact_medical_keyword_match(self):
        chunks = [
            make_chunk("c1", "高血压患者需要规律监测血压，减少钠盐摄入。"),
            make_chunk("c2", "二甲双胍常用于糖尿病治疗，常见用药风险包括胃肠道反应。"),
            make_chunk("c3", "肺炎常见症状包括发热、咳嗽、咳痰。"),
        ]
        index = BM25TextIndex(chunks)

        results = index.search("二甲双胍有哪些用药风险", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk"].chunk_id, "c2")
        self.assertGreater(results[0]["score"], 0.0)

    def test_bm25_returns_empty_results_when_query_has_no_overlap(self):
        chunks = [make_chunk("c1", "高血压患者需要规律监测血压。")]
        index = BM25TextIndex(chunks)

        results = index.search("unrelated-token", top_k=2)

        self.assertEqual(results, [])


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_rrf_merges_vector_and_bm25_rankings_by_chunk_id(self):
        chunk_a = make_chunk("a", "高血压日常管理。")
        chunk_b = make_chunk("b", "糖尿病血糖管理。")
        chunk_c = make_chunk("c", "二甲双胍用药风险。")

        fused = reciprocal_rank_fusion(
            [
                [{"chunk": chunk_a, "score": 0.90}, {"chunk": chunk_b, "score": 0.70}],
                [{"chunk": chunk_c, "score": 5.20}, {"chunk": chunk_a, "score": 3.10}],
            ],
            top_k=3,
            rank_constant=20,
        )

        chunk_ids = [item["chunk"].chunk_id for item in fused]
        self.assertEqual(chunk_ids[0], "a")
        self.assertCountEqual(chunk_ids, ["a", "b", "c"])
        self.assertIn("component_scores", fused[0])


class MedicalRetrieverHybridTests(unittest.TestCase):
    def test_hybrid_retrieve_keeps_result_shape_and_adds_keyword_only_hit(self):
        vector_chunk = make_chunk("vector", "高血压患者需要控制体重。")
        keyword_chunk = make_chunk("keyword", "二甲双胍常见用药风险包括胃肠道反应。")

        class FakeEmbedder:
            def encode(self, texts):
                return [[1.0, 0.0]]

        class FakeStore:
            chunks = [vector_chunk, keyword_chunk]

            def search(self, query_embedding, top_k=4):
                return [{"chunk": vector_chunk, "score": 0.95}]

        retriever = MedicalRetriever.__new__(MedicalRetriever)
        retriever.embedder = FakeEmbedder()
        retriever.store = FakeStore()
        retriever.top_k = 2
        retriever.retrieval_mode = "hybrid"
        retriever.bm25_top_k = 4
        retriever.rrf_rank_constant = 20
        retriever.bm25_index = BM25TextIndex(FakeStore.chunks)

        results = retriever.retrieve("二甲双胍有哪些用药风险", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertCountEqual([item["chunk"].chunk_id for item in results], ["vector", "keyword"])
        self.assertTrue(all("score" in item and "chunk" in item for item in results))


if __name__ == "__main__":
    unittest.main()
