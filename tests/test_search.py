"""Expansion SKOS, BM25 sur les enfants, remontée aux parents et reranking, sans modèle."""

import json
import tempfile
import unittest
from pathlib import Path

import bm25s
import numpy as np
from langchain_core.documents import Document

from docling_hybrid_rag.expansion import expand_query, load_thesaurus
from docling_hybrid_rag.indexing import LEXICAL_SETTINGS, lexical_tokens, weighted_scores
from docling_hybrid_rag.retrieval import KnowledgeBase, encode_question, hybrid_retrieval, rank_parents, retrieve_bm25

THESAURUS = {"language": "fr", "concepts": [
    {"id": "ibnr", "prefLabel": "IBNR", "altLabel": ["sinistres survenus non déclarés"],
     "hiddenLabel": ["incurred but not reported"], "broader": ["provisionnement"], "narrower": [], "related": []},
    {"id": "provisionnement", "prefLabel": "provisionnement", "altLabel": [], "hiddenLabel": [],
     "broader": [], "narrower": ["ibnr"], "related": []},
]}


def knowledge_base(child_texts):
    """Un parent par paire d'enfants ; l'index BM25 suit les réglages du package."""
    children = [Document(page_content=text, metadata={"child_id": f"c{i}", "parent_id": f"p{i // 2}"})
                for i, text in enumerate(child_texts)]
    parents = [Document(page_content="", metadata={"parent_id": f"p{i}"}) for i in range((len(children) + 1) // 2)]
    index = bm25s.BM25()
    index.index(lexical_tokens(child_texts), show_progress=False)
    vectors = np.zeros((len(children), 1024), dtype=np.float32)
    vectors[:, 0] = 1
    return KnowledgeBase(Path("."), parents, children, vectors, index, {}, LEXICAL_SETTINGS)


class ExpansionTests(unittest.TestCase):
    def test_without_thesaurus_the_query_is_unchanged(self):
        self.assertEqual(expand_query("Qu'est-ce que l'IBNR ?"),
                         [{"term": "Qu'est-ce que l'IBNR ?", "source": "original", "weight": 1.0}])

    def test_recognized_concept_adds_its_labels_but_not_its_relations_by_default(self):
        terms = expand_query("Comment estimer les sinistres survenus non déclarés ?", THESAURUS)
        self.assertEqual([(t["term"], t["source"], t["weight"]) for t in terms[1:]], [("IBNR", "prefLabel", 0.95)])

    def test_hidden_label_recognizes_without_being_added(self):
        terms = expand_query("incurred but not reported claims", THESAURUS)
        self.assertEqual({t["term"] for t in terms[1:]}, {"IBNR", "sinistres survenus non déclarés"})

    def test_relations_are_added_only_when_the_policy_names_them(self):
        terms = expand_query("IBNR", THESAURUS, policy={"prefLabel": 0.95, "altLabel": 0.85, "broader": 0.4})
        self.assertIn(("provisionnement", "broader", 0.4), [(t["term"], t["source"], t["weight"]) for t in terms])

    def test_a_label_inside_another_word_is_not_recognized(self):
        self.assertEqual(len(expand_query("IBNRX", THESAURUS)), 1)

    def test_thesaurus_is_read_from_a_json_path_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thesaurus.json"
            path.write_text(json.dumps(THESAURUS), encoding="utf-8")
            self.assertEqual(load_thesaurus(path), THESAURUS)
        with self.assertRaisesRegex(ValueError, "concepts"):
            load_thesaurus({"concepts": [{"id": "sans-libelle"}]})


class LexicalSearchTests(unittest.TestCase):
    def test_original_tokens_alone_score_like_plain_bm25(self):
        base = knowledge_base(["méthode de Mack et variance", "Chain-Ladder et facteurs", "Mack encore Mack"])
        tokens = lexical_tokens("variance de Mack", return_ids=False)[0]
        expected = base.bm25_index.get_scores(tokens)
        scores = weighted_scores(base.bm25_index, expand_query("variance de Mack"))
        np.testing.assert_allclose(scores, expected)

    def test_stemming_matches_singular_and_plural(self):
        base = knowledge_base(["Les provisions techniques", "Tarification automobile"])
        self.assertEqual([hit["child_id"] for hit in retrieve_bm25("provision", base)], ["c0"])

    def test_expansion_finds_a_passage_written_with_the_synonym(self):
        base = knowledge_base(["Estimation des IBNR par triangle", "Tarification automobile",
                               "Gestion des contrats", "Réassurance proportionnelle"])
        # Aucun mot de la question, même racinisé, ne figure dans le passage visé.
        question = "Comment évaluer les sinistres survenus non déclarés ?"
        self.assertEqual(retrieve_bm25(question, base), [])
        hits = retrieve_bm25(expand_query(question, THESAURUS), base)
        self.assertEqual([hit["child_id"] for hit in hits], ["c0"])


class QuestionVectorCacheTests(unittest.TestCase):
    def test_a_question_is_encoded_once_then_read_from_the_base(self):
        class CountingEncoder:
            calls = 0

            def encode(self, text, **kwargs):
                CountingEncoder.calls += 1
                vector = np.zeros(1024, dtype=np.float32)
                vector[0] = 1
                return vector

        with tempfile.TemporaryDirectory() as directory:
            base = knowledge_base(["Mack variance", "Chain-Ladder"])
            base.knowledge_dir = Path(directory)
            first = encode_question("Qu'est-ce que l'IBNR ?", base, encoder=CountingEncoder())
            second = encode_question("Qu'est-ce que l'IBNR ?", base, encoder=CountingEncoder())
            np.testing.assert_array_equal(first, second)
            self.assertEqual(CountingEncoder.calls, 1)
            encode_question("Une autre question ?", base, encoder=CountingEncoder())
            self.assertEqual(CountingEncoder.calls, 2)


class ParentRankingTests(unittest.TestCase):
    def test_each_channel_keeps_the_best_child_of_each_parent(self):
        base = knowledge_base(["Mack variance", "Mack hypothèses", "Chain-Ladder", "facteurs"])
        hits = [{"child_id": "c1", "score": 3}, {"child_id": "c0", "score": 2}, {"child_id": "c2", "score": 1}]
        self.assertEqual(rank_parents(hits, base.children_by_id),
                         [{"parent_id": "p0", "child_id": "c1", "score": 3}, {"parent_id": "p1", "child_id": "c2", "score": 1}])

    def test_reranker_reorders_parents_from_their_best_children(self):
        base = knowledge_base(["Mack variance des provisions", "Mack hypothèses",
                               "Chain-Ladder variance", "Chain-Ladder facteurs"])

        class Reranker:
            def predict(self, pairs, **kwargs):
                # Le passage sur les facteurs est jugé le plus pertinent.
                return np.array([0.9 if "facteurs" in passage else 0.1 for _, passage in pairs])

        result = hybrid_retrieval("variance Mack facteurs", base, question_vector=base.vectors[0],
                                  reranker=Reranker(), context_k=2)
        self.assertEqual(result["parent_ids"][0], "p1")
        self.assertEqual(result["best_score"], 0.9)
        self.assertEqual({hit["parent_id"] for hit in result["reranked"]}, {"p0", "p1"})

    def test_without_reranker_the_rrf_order_is_kept(self):
        base = knowledge_base(["Mack variance", "Mack", "Chain-Ladder", "facteurs"])
        result = hybrid_retrieval("Mack variance", base, question_vector=base.vectors[0], context_k=2)
        self.assertIsNone(result["reranked"])
        self.assertEqual(result["parent_ids"][0], "p0")


if __name__ == "__main__":
    unittest.main()
