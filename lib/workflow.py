"""Question -> réponse directe ou recherche -> contexte -> réponse sourcée."""

from pathlib import Path
from typing import TypedDict

from langgraph.graph import START, END, StateGraph

from lib.context import build_context
from lib.retrieval import hybrid_retrieval, load_knowledge_base
from lib.settings import get_llm
from lib.answer import generate_answer, sourced_answer_markdown


class RagState(TypedDict, total=False):
    question: str
    route: str
    retrieval: dict
    context: dict
    sources: dict
    answer: str


def needs_evidence(question):
    # Seules quelques salutations explicites évitent la recherche documentaire.
    normalized = question.casefold().strip().rstrip(".!?").strip()
    return normalized not in {"bonjour", "salut", "bonsoir", "merci"}


def create_workflow(data_dir: str | Path = "data", *, env_path: str | Path = ".env",
                    model: str | None = None, llm=None, encoder=None):
    """Créer un graphe appelable avec invoke({'question': '...'}).

    La base est chargée à la première question documentaire, puis réutilisée.
    Après une nouvelle ingestion, recréer le graphe pour charger les nouveaux index.
    """
    knowledge = None

    def route_question(state):
        if not isinstance(state.get("question"), str) or not state["question"].strip():
            raise ValueError("La question doit être une chaîne non vide.")
        return "retrieve" if needs_evidence(state["question"]) else "direct"

    def direct(state):
        # Aucune clé, aucun index et aucun modèle ne sont nécessaires pour une salutation.
        return {"route": "direct", "answer": "Bonjour ! Je peux vous aider à lire les documents de la base.",
                "retrieval": {}, "context": {"parents": [], "sources": {}}, "sources": {}}

    def retrieve(state):
        nonlocal knowledge
        if knowledge is None:
            knowledge = load_knowledge_base(data_dir)
        hits = hybrid_retrieval(state["question"], knowledge, encoder=encoder)
        return {"route": "rag", "retrieval": hits}

    def prepare_context(state):
        return {"context": build_context(state["retrieval"]["parent_ids"], knowledge)}

    def answer(state):
        context = state["context"]
        response = generate_answer(state["question"], context,
                                   llm=llm or get_llm(env_path, model))
        cited = {key for paragraph in response.paragraphs for key in paragraph.source_ids}
        return {"answer": sourced_answer_markdown(response, context["sources"]),
                "sources": {key: context["sources"][key] for key in cited}}

    graph = StateGraph(RagState)
    graph.add_node("direct", direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("context", prepare_context)
    graph.add_node("answer", answer)
    graph.add_conditional_edges(START, route_question, {"direct": "direct", "retrieve": "retrieve"})
    graph.add_edge("direct", END)
    graph.add_edge("retrieve", "context")
    graph.add_edge("context", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
