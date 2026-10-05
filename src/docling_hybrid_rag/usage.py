"""Jetons consommés par chaque appel au modèle de langage.

Une application qui paie ses appels doit pouvoir les compter. Le workflow renvoie `result["usage"]` :
une entrée par appel (routeur, reformulation, réponse directe, réponse sourcée, contrôle de soutien).
"""


def usage_entry(node: str, message, model: str | None = None) -> dict:
    """Entrée d'usage d'un appel, lue dans `usage_metadata` du message du modèle (zéro si le fournisseur n'en donne pas)."""
    metadata = getattr(message, "usage_metadata", None) or {}
    response = getattr(message, "response_metadata", None) or {}
    return {"node": node, "model": model or response.get("model_name") or response.get("model"),
            "input_tokens": int(metadata.get("input_tokens", 0)), "output_tokens": int(metadata.get("output_tokens", 0)),
            "total_tokens": int(metadata.get("total_tokens", 0))}


def total_usage(entries: list[dict]) -> dict:
    """Somme des jetons d'une liste d'entrées, par exemple `total_usage(result["usage"])`."""
    return {key: sum(entry[key] for entry in entries) for key in ("input_tokens", "output_tokens", "total_tokens")}
