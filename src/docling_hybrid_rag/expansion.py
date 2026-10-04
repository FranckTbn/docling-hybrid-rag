"""Expansion de requête contrôlée par un thésaurus inspiré de SKOS.

Méthode de l'article « Query Expansion » (https://ornelle.quarto.pub/query-expansion/) :
la requête originale est conservée,
les libellés des concepts reconnus sont ajoutés avec un poids qui dépend de leur
source, puis chaque terme pèse sur le score BM25 en proportion de ce poids.

Le thésaurus est un JSON de la forme :

    {"language": "fr", "concepts": [
        {"id": "ibnr", "prefLabel": "IBNR",
         "altLabel": ["sinistres survenus non déclarés"],
         "hiddenLabel": ["incurred but not reported"],
         "broader": [], "narrower": [], "related": []}]}

Sans thésaurus, la requête reste inchangée.
"""

import json
import re
from pathlib import Path

# Par défaut, seuls les libellés du concept reconnu sont ajoutés : ils restent
# proches de son sens. Les relations ne sont pas des synonymes ; elles ne sont
# injectées que si la politique les cite, avec les poids réduits de l'article.
DEFAULT_POLICY = {"prefLabel": 0.95, "altLabel": 0.85}
RELATION_WEIGHTS = {"narrower": 0.65, "broader": 0.40, "related": 0.25}
MAX_ADDED_TERMS = 12


def load_thesaurus(thesaurus: dict | str | Path | None) -> dict | None:
    """Accepter un dictionnaire, un chemin vers un JSON, ou rien."""
    if thesaurus is None:
        return None
    data = thesaurus if isinstance(thesaurus, dict) else json.loads(Path(thesaurus).read_text(encoding="utf-8"))
    concepts = data.get("concepts")
    if not isinstance(concepts, list) or not all("id" in c and "prefLabel" in c for c in concepts):
        raise ValueError("Le thésaurus doit contenir une liste 'concepts' avec 'id' et 'prefLabel'.")
    return data


def _normalize(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _is_present(label: str, normalized_query: str) -> bool:
    # Un libellé doit apparaître comme expression complète, pas à l'intérieur d'un mot.
    return re.search(rf"(?<!\w){re.escape(_normalize(label))}(?!\w)", normalized_query) is not None


def expand_query(query: str, thesaurus: dict | str | Path | None = None,
                 policy: dict[str, float] | None = None,
                 max_added_terms: int = MAX_ADDED_TERMS) -> list[dict]:
    """Renvoyer la requête originale (poids 1) puis les termes ajoutés, avec leur source."""
    terms = [{"term": query, "source": "original", "weight": 1.0}]
    thesaurus = load_thesaurus(thesaurus)
    if thesaurus is None:
        return terms

    policy = DEFAULT_POLICY if policy is None else policy
    normalized_query = _normalize(query)
    concepts = {concept["id"]: concept for concept in thesaurus["concepts"]}

    def labels(concept):
        return [concept["prefLabel"], *concept.get("altLabel", [])]

    # hiddenLabel sert seulement à reconnaître un concept, jamais à l'ajouter.
    recognized = [
        concept for concept in concepts.values()
        if any(_is_present(label, normalized_query)
               for label in [*labels(concept), *concept.get("hiddenLabel", [])])
    ]

    added: dict[str, dict] = {}

    def add(term, source):
        # Une formulation déjà présente dans la requête n'est pas renforcée.
        if source not in policy or _is_present(term, normalized_query):
            return
        key = _normalize(term)
        if key not in added or policy[source] > added[key]["weight"]:
            added[key] = {"term": " ".join(str(term).split()), "source": source, "weight": policy[source]}

    for concept in recognized:
        add(concept["prefLabel"], "prefLabel")
        for label in concept.get("altLabel", []):
            add(label, "altLabel")
        for relation in RELATION_WEIGHTS:
            for target in concept.get(relation, []):
                if target in concepts:
                    for label in labels(concepts[target]):
                        add(label, relation)

    # Plafonner les ajouts limite la dérive de la requête ; le tri reste reproductible.
    ranked = sorted(added.values(), key=lambda item: (-item["weight"], item["term"].casefold()))
    return terms + ranked[:max_added_terms]
