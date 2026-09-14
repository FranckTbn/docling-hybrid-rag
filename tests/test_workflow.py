"""Routage simulé, vrai graphe et vrais checkpoints, sans coût API."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from lib.workflow import RouteDecision, create_workflow


def conversation(name):
    return {"configurable": {"thread_id": name}}


def decision(route, budget=3, question="Question autonome"):
    return {"route": route, "context_k": budget, "search_question": question, "reason": "Motif simulé."}


class ScriptedLLM:
    """Les décisions sont imposées par le test, jamais évaluées comme un vrai modèle."""
    def __init__(self, routes, direct_answers=(), answers=()):
        self.routes, self.direct_answers, self.answers = list(routes), list(direct_answers), list(answers)
        self.routing_calls, self.direct_calls, self.answer_calls = [], [], []

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            if schema is RouteDecision:
                self.routing_calls.append(messages)
                parsed = self.routes.pop(0)
            else:
                self.answer_calls.append(messages)
                parsed = self.answers.pop(0)
            return {"parsed": parsed, "parsing_error": None, "raw": AIMessage(content="")}
        return SimpleNamespace(invoke=invoke)

    def invoke(self, messages):
        self.direct_calls.append(messages)
        return AIMessage(content=self.direct_answers.pop(0))


class WorkflowTests(unittest.TestCase):
    def test_direct_uses_llm_decision_and_response_without_loading_index(self):
        llm = ScriptedLLM([decision("direct")], ["Voici une aide pour formuler votre question."])
        with patch("lib.workflow.load_knowledge_base", side_effect=AssertionError("lecture interdite")):
            result = create_workflow(llm=llm).invoke({"question": "Aide-moi à formuler ma question"}, conversation("a"))
        self.assertEqual(result["route"], "direct")
        self.assertEqual(result["context_k"], 0)
        self.assertEqual(len(llm.routing_calls), 1)
        self.assertEqual(len(llm.direct_calls), 1)
        self.assertEqual(result["answer"], "Voici une aide pour formuler votre question.")
        self.assertEqual(result["sources"], {})

    def test_router_budget_and_rewritten_question_reach_retrieval_and_answer(self):
        from lib.answer import SourcedAnswer
        for budget in (3, 5, 7):
            with self.subTest(budget=budget):
                llm = ScriptedLLM([decision("retrieve", budget, "Quelles sont les limites de Mack ?")])
                graph = create_workflow(llm=llm)
                parent_ids = [f"parent-{i}" for i in range(budget)]
                context = {"parents": [{"text": "Preuve"}], "sources": {}}
                with patch("lib.workflow.load_knowledge_base", return_value=object()), \
                     patch("lib.workflow.hybrid_retrieval", return_value={"parent_ids": parent_ids}) as retrieve, \
                     patch("lib.workflow.build_context", return_value=context) as build, \
                     patch("lib.workflow.generate_answer", return_value=SourcedAnswer(
                         paragraphs=[], missing_information="Preuves insuffisantes.")) as generate:
                    result = graph.invoke({"question": "Et ses limites ?"}, conversation(str(budget)))
                self.assertEqual(retrieve.call_args.args[0], "Quelles sont les limites de Mack ?")
                self.assertEqual(retrieve.call_args.kwargs["context_k"], budget)
                self.assertEqual(build.call_args.args[0], parent_ids)
                self.assertEqual(generate.call_args.args[0], "Quelles sont les limites de Mack ?")
                self.assertEqual(result["context_k"], budget)
                self.assertEqual(result["messages"][-1].content, result["answer"])

    def test_last_three_messages_are_persisted_and_used_on_follow_up(self):
        llm = ScriptedLLM([decision("direct")] * 3, ["Bonjour Alice", "Je peux vous aider", "Je ne retrouve plus votre prénom"])
        graph, config = create_workflow(llm=llm), conversation("alice")
        graph.invoke({"question": "Je suis Alice"}, config)
        graph.invoke({"question": "Que peux-tu faire ?"}, config)
        result = graph.invoke({"question": "Quel est mon prénom ?"}, config)
        self.assertEqual([m.content for m in llm.routing_calls[-1][1:]],
                         ["Que peux-tu faire ?", "Je peux vous aider", "Quel est mon prénom ?"])
        self.assertEqual([m.content for m in llm.direct_calls[-1][1:]],
                         [m.content for m in llm.routing_calls[-1][1:]])
        self.assertEqual([m.content for m in result["messages"]],
                         ["Je peux vous aider", "Quel est mon prénom ?", "Je ne retrouve plus votre prénom"])
        self.assertEqual(len(graph.get_state(config).values["messages"]), 3)
        # La réponse simulée n'est pas un test de fidélité : le prénom est déjà hors de la fenêtre.
        self.assertNotIn("Je suis Alice", [m.content for m in llm.routing_calls[-1]])

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

    def test_invalid_route_or_budget_is_rejected_before_retrieval(self):
        for invalid in (decision("other"), decision("retrieve", 4), decision("retrieve", 0)):
            with self.subTest(invalid=invalid), patch("lib.workflow.hybrid_retrieval") as retrieve:
                with self.assertRaises(ValidationError):
                    create_workflow(llm=ScriptedLLM([invalid])).invoke({"question": "Question"}, conversation("bad"))
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
