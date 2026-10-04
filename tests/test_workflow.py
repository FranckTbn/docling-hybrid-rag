"""Routage simulé, vrai graphe et vrais checkpoints, sans coût API."""

import threading
import time
import unittest
from html import unescape
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from docling_hybrid_rag.answer import SourcedAnswer
from docling_hybrid_rag.verification import SupportReport, check_support
from docling_hybrid_rag.workflow import RewriteDecision, RouteDecision, create_workflow, split_budget


def conversation(name):
    return {"configurable": {"thread_id": name}}


def decision(route, budget=3, question="Question autonome", sub_questions=None):
    return {"route": route, "context_k": budget, "search_question": question,
            "sub_questions": sub_questions or [question], "reason": "Motif simulé."}


def supported(paragraphs):
    return {"checks": [{"paragraph": number, "verdict": "soutenu", "problem": ""}
                       for number in range(1, paragraphs + 1)], "alerts": [], "suggestions": []}


class ScriptedLLM:
    """Les décisions sont imposées par le test, jamais évaluées comme un vrai modèle."""
    def __init__(self, routes, direct_answers=(), answers=(), rewrites=(), reports=()):
        self.queues = {RouteDecision: list(routes), RewriteDecision: list(rewrites),
                       SupportReport: list(reports), "answer": list(answers)}
        self.calls = {key: [] for key in self.queues}
        self.direct_answers, self.direct_calls = list(direct_answers), []

    @property
    def routing_calls(self):
        return self.calls[RouteDecision]

    def with_structured_output(self, schema, **kwargs):
        # Le schéma de réponse est un dictionnaire JSON ; les autres sont des classes Pydantic.
        key = schema if isinstance(schema, type) and schema in self.queues else "answer"

        def invoke(messages):
            self.calls[key].append(messages)
            return {"parsed": self.queues[key].pop(0), "parsing_error": None, "raw": AIMessage(content="")}
        return SimpleNamespace(invoke=invoke)

    def invoke(self, messages):
        self.direct_calls.append(messages)
        return AIMessage(content=self.direct_answers.pop(0))


class UnitEncoder:
    """Un vecteur normé par texte ; la valeur ne compte pas, seul le passage du vecteur est testé."""
    def encode(self, texts, **kwargs):
        rows = np.zeros((1 if isinstance(texts, str) else len(texts), 1024), dtype=np.float32)
        rows[:, 0] = 1
        return rows[0] if isinstance(texts, str) else rows


def search_result(parent_ids, best_score=None):
    return {"parent_ids": parent_ids, "best_score": best_score,
            "expansion": [{"term": "q", "source": "original", "weight": 1.0}]}


def context_for(parent_ids):
    return {"parents": [{"parent_id": pid, "text": f"Texte {pid}", "images": []} for pid in parent_ids],
            "sources": {}}


class WorkflowTests(unittest.TestCase):
    def run_retrieval(self, llm, question, *, search, contexts=context_for, answer=None, thread="t", **options):
        graph = create_workflow(llm=llm, encoder=UnitEncoder(), **options)
        draft = answer or SourcedAnswer(paragraphs=[], missing_information="Preuves insuffisantes.")
        with patch("docling_hybrid_rag.workflow.load_knowledge_base", return_value=object()), \
             patch("docling_hybrid_rag.workflow.hybrid_retrieval", side_effect=search) as retrieve, \
             patch("docling_hybrid_rag.workflow.build_context", side_effect=lambda ids, kb: contexts(ids)) as build, \
             patch("docling_hybrid_rag.workflow.generate_answer", return_value=draft) as generate:
            result = graph.invoke({"question": question}, conversation(thread))
        return result, retrieve, build, generate

    def test_direct_uses_llm_decision_and_response_without_loading_index(self):
        llm = ScriptedLLM([decision("direct")], ["Voici une aide pour formuler votre question."])
        with patch("docling_hybrid_rag.workflow.load_knowledge_base", side_effect=AssertionError("lecture interdite")):
            result = create_workflow(llm=llm).invoke({"question": "Aide-moi à formuler ma question"}, conversation("a"))
        self.assertEqual(result["route"], "direct")
        self.assertEqual(result["context_k"], 0)
        self.assertEqual(result["sub_questions"], [])
        self.assertEqual(len(llm.direct_calls), 1)
        self.assertEqual(result["answer"], "Voici une aide pour formuler votre question.")

    def test_single_theme_keeps_the_whole_budget_and_the_rewritten_question(self):
        for budget in (3, 5, 7):
            with self.subTest(budget=budget):
                llm = ScriptedLLM([decision("retrieve", budget, "Quelles sont les limites de Mack ?")])
                ids = [f"p{i}" for i in range(budget)]
                result, retrieve, build, generate = self.run_retrieval(
                    llm, "Et ses limites ?", search=lambda *a, **k: search_result(ids), thread=str(budget))
                self.assertEqual(retrieve.call_args.args[0], "Quelles sont les limites de Mack ?")
                self.assertEqual(retrieve.call_args.kwargs["context_k"], budget)
                self.assertEqual(retrieve.call_args.kwargs["question_vector"].shape, (1024,))
                self.assertEqual(build.call_args.args[0], ids)
                self.assertEqual(generate.call_args.args[0], "Quelles sont les limites de Mack ?")
                self.assertEqual(result["messages"][-1].content, result["answer"])

    def test_each_sub_question_has_its_own_search_budget_and_sources(self):
        questions = ["Comment fonctionne Chain-Ladder ?", "Comment fonctionne la méthode de Mack ?"]
        by_query = {questions[0]: ["cl-1", "commun", "cl-2"], questions[1]: ["mack-1", "commun"]}
        llm = ScriptedLLM([decision("retrieve", 5, "Compare Chain-Ladder et Mack", questions)])
        result, retrieve, build, generate = self.run_retrieval(
            llm, "Compare Chain-Ladder et Mack",
            search=lambda query, kb, **k: search_result(by_query[query][:k["context_k"]]))
        budgets = {call.args[0]: call.kwargs["context_k"] for call in retrieve.call_args_list}
        self.assertEqual(budgets, {questions[0]: 3, questions[1]: 2})
        # Un parent commun n'est transmis qu'une fois, avec ses deux thèmes.
        self.assertEqual(build.call_args.args[0], ["cl-1", "commun", "cl-2", "mack-1"])
        themes = generate.call_args.kwargs["themes"]
        self.assertEqual([theme["sub_question"] for theme in themes], questions)
        shared = next(p for p in result["context"]["parents"] if p["parent_id"] == "commun")
        self.assertEqual(shared["themes"], questions)

    def test_parallel_branches_never_use_the_shared_models_at_the_same_time(self):
        # Régression : les tokenizers Hugging Face lèvent « Already borrowed » si deux
        # branches appellent le même reranker en même temps.
        active, overlaps, lock = [0], [], threading.Lock()

        def guarded_search(query, kb, **kwargs):
            with lock:
                active[0] += 1
                overlaps.append(active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return search_result([f"{query}-p"])

        llm = ScriptedLLM([decision("retrieve", 7, "Compare A, B et C", ["A ?", "B ?", "C ?"])])
        self.run_retrieval(llm, "Compare A, B et C", search=guarded_search)
        self.assertEqual(max(overlaps), 1)

    def test_low_relevance_rewrites_once_then_declares_the_theme_without_passage(self):
        llm = ScriptedLLM([decision("retrieve", 3, "Qu'est-ce que l'IBNR ?")], rewrites=[{"query": "sinistres survenus non déclarés"}])
        result, retrieve, build, generate = self.run_retrieval(
            llm, "Qu'est-ce que l'IBNR ?", search=lambda *a, **k: search_result(["p1"], best_score=0.1),
            reranker=object(), relevance_threshold=0.5)
        self.assertEqual([call.args[0] for call in retrieve.call_args_list],
                         ["Qu'est-ce que l'IBNR ?", "sinistres survenus non déclarés"])
        self.assertEqual(len(llm.calls[RewriteDecision]), 1)
        self.assertEqual(result["themes"][0]["status"], "not_found")
        self.assertEqual(build.call_args.args[0], [])
        generate.assert_not_called()
        self.assertIn("Aucun passage pertinent retrouvé pour : Qu'est-ce que l'IBNR ?", unescape(result["answer"]))

    def test_rewrite_that_finds_relevant_passages_is_kept(self):
        scores = iter([0.1, 0.9])
        llm = ScriptedLLM([decision("retrieve")], rewrites=[{"query": "Formulation métier"}])
        result, retrieve, *_ = self.run_retrieval(
            llm, "Question", search=lambda *a, **k: search_result(["p1"], best_score=next(scores)),
            reranker=object(), relevance_threshold=0.5)
        theme = result["themes"][0]
        self.assertEqual((theme["status"], theme["query"], theme["rewrites"]), ("found", "Formulation métier", ["Formulation métier"]))

    def test_support_alert_and_suggestion_become_part_of_the_message(self):
        sources = {key: {"parent_id": "p1", "name": name, "pages": [12], "text": text,
                         "document_title": "Guide", "source_url": "https://example.org/guide.pdf", "ref": None}
                   for key, name, text in (("s1", "Mack", "La MSEP est une erreur quadratique moyenne."),
                                           ("s2", "Chain-Ladder", "Facteurs de développement."))}
        answer = SourcedAnswer(paragraphs=[{"text": "Mack donne une variance.", "source_ids": ["s1"]},
                                           {"text": "Chain-Ladder utilise des facteurs.", "source_ids": ["s2"]}],
                               missing_information="")
        report = {"checks": [{"paragraph": 1, "verdict": "partiel", "problem": "La source parle d'erreur quadratique, pas de variance."},
                             {"paragraph": 2, "verdict": "soutenu", "problem": ""}],
                  "alerts": [], "suggestions": ["Relire la définition de la MSEP page 12."]}
        llm = ScriptedLLM([decision("retrieve")], reports=[report])
        result, *_ = self.run_retrieval(llm, "Question", search=lambda *a, **k: search_result(["p1"]),
                                        contexts=lambda ids: {"parents": context_for(ids)["parents"], "sources": sources},
                                        answer=answer)
        self.assertIn("Alerte de vérification", result["answer"])
        self.assertIn("Paragraphe 1, partiellement soutenu", result["answer"])
        self.assertIn("Relire la définition de la MSEP page 12.", result["answer"])
        self.assertEqual(result["messages"][-1].content, result["answer"])
        # Le vérificateur reçoit le texte des sources citées, pas seulement leurs identifiants.
        self.assertIn("erreur quadratique moyenne", llm.calls[SupportReport][0][1].content)

    def test_report_that_skips_a_paragraph_raises_an_alert(self):
        answer = SourcedAnswer(paragraphs=[{"text": "A", "source_ids": ["s"]}, {"text": "B", "source_ids": ["s"]}],
                               missing_information="")
        sources = {"s": {"name": "Section", "pages": [1], "text": "A et B"}}
        report = check_support("Q", answer, sources, llm=ScriptedLLM([], reports=[supported(1)]))
        self.assertEqual(report.alerts, ["Le contrôle n'a pas évalué le paragraphe 2."])

    def test_themes_are_reset_between_turns(self):
        llm = ScriptedLLM([decision("retrieve", 3, "Q1", ["Q1a", "Q1b"]), decision("retrieve", 3, "Q2")])
        graph = create_workflow(llm=llm, encoder=UnitEncoder())
        with patch("docling_hybrid_rag.workflow.load_knowledge_base", return_value=object()), \
             patch("docling_hybrid_rag.workflow.hybrid_retrieval", side_effect=lambda *a, **k: search_result(["p"])), \
             patch("docling_hybrid_rag.workflow.build_context", side_effect=lambda ids, kb: context_for(ids)), \
             patch("docling_hybrid_rag.workflow.generate_answer",
                   return_value=SourcedAnswer(paragraphs=[], missing_information="Rien.")):
            graph.invoke({"question": "Q1"}, conversation("reset"))
            second = graph.invoke({"question": "Q2"}, conversation("reset"))
        self.assertEqual([theme["sub_question"] for theme in second["themes"]], ["Q2"])

    def test_threshold_without_reranker_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reranker"):
            create_workflow(llm=ScriptedLLM([]), relevance_threshold=0.5)

    def test_budget_is_split_without_leaving_a_theme_empty(self):
        self.assertEqual(split_budget(3, 1), [3])
        self.assertEqual(split_budget(5, 2), [3, 2])
        self.assertEqual(split_budget(7, 3), [3, 2, 2])
        self.assertEqual(split_budget(3, 3), [1, 1, 1])

    def test_last_three_messages_are_persisted_and_used_on_follow_up(self):
        llm = ScriptedLLM([decision("direct")] * 3, ["Bonjour Alice", "Je peux vous aider", "Je ne retrouve plus votre prénom"])
        graph, config = create_workflow(llm=llm), conversation("alice")
        graph.invoke({"question": "Je suis Alice"}, config)
        graph.invoke({"question": "Que peux-tu faire ?"}, config)
        result = graph.invoke({"question": "Quel est mon prénom ?"}, config)
        self.assertEqual([m.content for m in llm.routing_calls[-1][1:]],
                         ["Que peux-tu faire ?", "Je peux vous aider", "Quel est mon prénom ?"])
        self.assertEqual([m.content for m in result["messages"]],
                         ["Je peux vous aider", "Quel est mon prénom ?", "Je ne retrouve plus votre prénom"])
        self.assertEqual(len(graph.get_state(config).values["messages"]), 3)

    def test_threads_and_new_graph_have_separate_memories(self):
        llm = ScriptedLLM([decision("direct")] * 4, ["A", "B", "A suite", "Nouveau"])
        graph = create_workflow(llm=llm)
        graph.invoke({"question": "Je suis Alice"}, conversation("a"))
        graph.invoke({"question": "Je suis Bob"}, conversation("b"))
        self.assertEqual([m.content for m in llm.routing_calls[1][1:]], ["Je suis Bob"])
        graph.invoke({"question": "Mon prénom ?"}, conversation("a"))
        self.assertEqual([m.content for m in llm.routing_calls[2][1:]], ["Je suis Alice", "A", "Mon prénom ?"])
        create_workflow(llm=llm).invoke({"question": "Nouvelle session"}, conversation("a"))
        self.assertEqual([m.content for m in llm.routing_calls[3][1:]], ["Nouvelle session"])

    def test_invalid_route_budget_or_sub_questions_are_rejected_before_retrieval(self):
        invalid = (decision("other"), decision("retrieve", 4), decision("retrieve", 0),
                   {**decision("retrieve"), "sub_questions": []},
                   {**decision("retrieve"), "sub_questions": ["a", "b", "c", "d"]})
        for route in invalid:
            with self.subTest(route=route), patch("docling_hybrid_rag.workflow.hybrid_retrieval") as retrieve:
                with self.assertRaises(ValidationError):
                    create_workflow(llm=ScriptedLLM([route])).invoke({"question": "Question"}, conversation("bad"))
                retrieve.assert_not_called()

    def test_empty_question_and_missing_thread_id_do_not_call_model(self):
        llm = ScriptedLLM([])
        graph = create_workflow(llm=llm)
        with self.assertRaises(ValueError):
            graph.invoke({"question": "   "}, conversation("empty"))
        with self.assertRaises(ValueError):
            graph.invoke({"question": "Bonjour"})
        self.assertEqual(llm.routing_calls, [])


if __name__ == "__main__":
    unittest.main()
