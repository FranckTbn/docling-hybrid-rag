"""ViDoRe V3 : le benchmark de recherche dans des rapports techniques, mesuré page par page.

ViDoRe V3 (ILLUIN Technology et NVIDIA, 2026, https://arxiv.org/abs/2601.08620) rassemble des documents
d'entreprise réels (rapports financiers, manuels techniques, cours, rapports d'agences), des questions
écrites ou vérifiées par des annotateurs, et, pour chacune, les pages pertinentes notées 1 (utile) ou 2
(indispensable). Il se mesure en nDCG@10 et son classement est tenu à jour sur le tableau MTEB. Les
documents sont fournis en PDF (licence CC BY 4.0) : le pipeline complet s'exécute donc de bout en bout,
du parsing à la recherche.

    python -m docling_hybrid_rag.benchmarks.vidore_v3 download --dataset hr
    python -m docling_hybrid_rag.benchmarks.vidore_v3 baseline --dataset hr    # BM25 sur le texte fourni : vérifie le protocole
    python -m docling_hybrid_rag.benchmarks.vidore_v3 ingest   --dataset hr    # long : parse chaque PDF avec Docling
    python -m docling_hybrid_rag.benchmarks.vidore_v3 evaluate --dataset hr

Chaque étape reprend où elle s'est arrêtée. Les fichiers vont dans `--data-dir` (par défaut
`data/benchmarks/vidore-v3`) et la base dans `<data-dir>/<jeu>/knowledge`. Le module demande
`pip install "docling-hybrid-rag[benchmarks]"` (pyarrow lit les fichiers du jeu).
"""

import argparse
import json
import time
from pathlib import Path

# Révisions figées des huit jeux publics : tout le monde mesure sur les mêmes fichiers.
DATASETS = {
    "hr": {"revision": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3", "language": "en", "column": "H.R.",
           "title": "Rapports européens sur l'emploi (14 PDF, 1 110 pages)"},
    "finance_en": {"revision": "7f432c176d82e27546501ad8064a713ac3071809", "language": "en", "column": "Fin.",
                   "title": "Rapports annuels américains (6 PDF, 2 942 pages)"},
    "industrial": {"revision": "e26c864724f5dd71a3d7d739272d95637764cee9", "language": "en", "column": "Ind.",
                   "title": "Manuels techniques de l'US Air Force (27 PDF, 5 244 pages)"},
    "pharmaceuticals": {"revision": "3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c", "language": "en", "column": "Phar.",
                        "title": "Rapports de la FDA (52 PDF, 2 313 pages)"},
    "computer_science": {"revision": "d5cc75883d92e294f0c0fc2662551c9708a06ebc", "language": "en", "column": "C.S.",
                         "title": "Manuels d'informatique (2 PDF, 1 360 pages)"},
    "energy": {"revision": "caec06d3c73434d635f710f93bcd898331c59f20", "language": "fr", "column": "Ener.",
               "title": "Rapports publics français sur l'énergie (42 PDF, 2 229 pages)"},
    "physics": {"revision": "a0de276f515acc044b72cae8de53a44bb5a8f1f5", "language": "fr", "column": "Phys.",
                "title": "Cours de physique en français (42 PDF, 1 674 pages)"},
    "finance_fr": {"revision": "1d808daa08032ffecdf62da151a7f7a8fe2bd0c9", "language": "fr", "column": "Fin.",
                   "title": "Rapports annuels français (5 PDF, 2 384 pages)"},
}
LANGUAGE_NAMES = {"en": "english", "fr": "french"}

# nDCG@10 (×100) publiés dans l'article de ViDoRe V3, tableaux 9 (requêtes anglaises sur documents anglais)
# et 10 (requêtes françaises sur documents français), avec le texte des pages extrait des PDF.
PUBLISHED = {
    "english": {  # colonnes : C.S., Fin., Phar., H.R., Ind.
        "Jina-v4 (texte)": {"C.S.": 67.3, "Fin.": 56.5, "Phar.": 59.0, "H.R.": 58.8, "Ind.": 45.8},
        "Qwen3-Embedding-8B (texte)": {"C.S.": 73.5, "Fin.": 54.8, "Phar.": 62.4, "H.R.": 52.3, "Ind.": 45.3},
        "LFM2-ColBERT-350M (texte)": {"C.S.": 70.6, "Fin.": 48.3, "Phar.": 62.1, "H.R.": 53.2, "Ind.": 47.9},
        "BM25S (texte)": {"C.S.": 64.7, "Fin.": 49.9, "Phar.": 56.9, "H.R.": 49.6, "Ind.": 45.6},
        "BGE-M3 (texte)": {"C.S.": 63.6, "Fin.": 43.9, "Phar.": 54.7, "H.R.": 45.3, "Ind.": 39.0},
        "ColPali (images)": {"C.S.": 72.5, "Fin.": 43.3, "Phar.": 57.7, "H.R.": 53.3, "Ind.": 47.0},
        "Jina-v4 (images)": {"C.S.": 74.2, "Fin.": 66.1, "Phar.": 65.2, "H.R.": 64.6, "Ind.": 55.9},
        "ColEmbed-3B-v2 (images)": {"C.S.": 78.6, "Fin.": 69.1, "Phar.": 67.6, "H.R.": 65.4, "Ind.": 56.8},
    },
    "french": {  # colonnes : Phys., Ener., Fin.
        "Jina-v4 (texte)": {"Phys.": 44.0, "Ener.": 63.4, "Fin.": 44.8},
        "Qwen3-Embedding-8B (texte)": {"Phys.": 45.8, "Ener.": 60.2, "Fin.": 37.6},
        "BM25S (texte)": {"Phys.": 39.8, "Ener.": 57.4, "Fin.": 35.9},
        "BGE-M3 (texte)": {"Phys.": 38.3, "Ener.": 53.1, "Fin.": 28.4},
        "ColPali (images)": {"Phys.": 43.2, "Ener.": 50.5, "Fin.": 23.6},
        "Jina-v4 (images)": {"Phys.": 46.8, "Ener.": 66.7, "Fin.": 48.6},
        "ColEmbed-3B-v2 (images)": {"Phys.": 48.2, "Ener.": 67.5, "Fin.": 48.2},
    },
}
PAPER = "https://arxiv.org/abs/2601.08620"


def _parquet(path: Path, columns: list[str] | None = None):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError('Ce benchmark lit des fichiers parquet : pip install "docling-hybrid-rag[benchmarks]"') from error
    return parquet.read_table(path, columns=columns).to_pandas()


def dataset_dir(data_dir: Path, dataset: str) -> Path:
    return Path(data_dir) / dataset


def download(dataset: str, data_dir: Path) -> Path:
    """Télécharger le jeu à sa révision figée : PDF, questions, notes de pertinence et table des pages."""
    from huggingface_hub import snapshot_download

    spec = DATASETS[dataset]
    target = dataset_dir(data_dir, dataset)
    snapshot_download(repo_id=f"vidore/vidore_v3_{dataset}", repo_type="dataset", revision=spec["revision"],
                      local_dir=target, allow_patterns=["pdfs/*", "queries/*", "qrels/*", "documents_metadata/*",
                                                        "corpus/*", "README.md"])
    return target


def load_documents(target: Path) -> dict[str, Path]:
    """Identifiant de document (celui du benchmark) vers son PDF."""
    metadata = _parquet(target / "documents_metadata" / "test-00000-of-00001.parquet")
    return {row.doc_id: target / "pdfs" / row.file_name for row in metadata.itertuples()}


def load_queries(target: Path, dataset: str) -> list[dict]:
    """Les questions de la langue du jeu, chacune avec ses pages pertinentes {(document, page à partir de 0): note}."""
    queries = _parquet(target / "queries" / "test-00000-of-00001.parquet")
    queries = queries[queries.language == LANGUAGE_NAMES[DATASETS[dataset]["language"]]]
    pages = _parquet(next((target / "corpus").glob("test-*.parquet")), ["corpus_id", "doc_id", "page_number_in_doc"])
    page_of = {row.corpus_id: (row.doc_id, int(row.page_number_in_doc)) for row in pages.itertuples()}
    relevance: dict[int, dict] = {}
    for row in _parquet(target / "qrels" / "test-00000-of-00001.parquet").itertuples():
        relevance.setdefault(row.query_id, {})[page_of[row.corpus_id]] = int(row.score)
    return [{"query_id": int(row.query_id), "question": row.query, "relevance": relevance[row.query_id],
             "query_types": list(row.query_types), "content_types": list(row.content_type)}
            for row in queries.itertuples() if row.query_id in relevance]


def ingest(dataset: str, data_dir: Path, knowledge_dir: Path | None = None, *, pdf_options=None,
           only: list[str] | None = None) -> None:
    """Parser, découper et indexer chaque PDF du jeu (le titre de chaque document est son identifiant)."""
    from docling_hybrid_rag import ingest_document

    target = dataset_dir(data_dir, dataset)
    knowledge_dir = knowledge_dir or target / "knowledge"
    documents = load_documents(target)
    log = target / "ingestion.jsonl"
    done = {json.loads(line)["document"] for line in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()
    for document, pdf in sorted(documents.items()):
        if (only and document not in only) or document in done:
            continue
        started = time.perf_counter()
        record = ingest_document(pdf, knowledge_dir, title=document, pdf_options=pdf_options,
                                 language=DATASETS[dataset]["language"])
        entry = {"document": document, "parents": record["parents"], "children": record["children"],
                 "seconds": round(time.perf_counter() - started, 1)}
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(entry, flush=True)


def baseline_bm25(dataset: str, data_dir: Path, k: int = 10) -> dict:
    """BM25 sur le texte des pages fourni avec le jeu : reproduit le BM25S publié et vérifie le protocole.

    Le texte vient de l'extraction des auteurs du benchmark, pas de notre parsing : si ce score retrouve
    celui de l'article, la lecture des questions, des notes et la formule du nDCG sont les bonnes.
    """
    import bm25s

    from docling_hybrid_rag.indexing import lexical_settings, lexical_tokens
    from docling_hybrid_rag.page_evaluation import ndcg_at_k

    target = dataset_dir(data_dir, dataset)
    pages = _parquet(next((target / "corpus").glob("test-*.parquet")), ["doc_id", "markdown", "page_number_in_doc"])
    keys = [(row.doc_id, int(row.page_number_in_doc)) for row in pages.itertuples()]
    settings = lexical_settings(DATASETS[dataset]["language"])
    index = bm25s.BM25()
    index.index(lexical_tokens(list(pages.markdown), settings), show_progress=False)
    scores = []
    for query in load_queries(target, dataset):
        tokens = lexical_tokens([query["question"]], settings, return_ids=False)[0]
        vector = index.get_scores(tokens)
        order = vector.argsort()[::-1][:100]
        scores.append(ndcg_at_k([keys[i] for i in order], query["relevance"], k))
    return {"questions": len(scores), f"ndcg@{k}": 100 * sum(scores) / len(scores)}


def summarize(rows: list[dict]) -> dict:
    from statistics import mean

    from docling_hybrid_rag.passage_evaluation import bootstrap_interval

    summary = {"questions": len(rows)}
    for key in (key for key in rows[0] if key != "query_id"):
        values = [float(row[key]) for row in rows]
        low, high = bootstrap_interval(values)
        summary[key] = {"mean": mean(values), "low": low, "high": high}
    return summary


def comparison(dataset: str, ours: dict[str, float]) -> list[dict]:
    """Nos scores et les scores publiés de la même colonne, du meilleur au moins bon."""
    spec = DATASETS[dataset]
    published = PUBLISHED["english" if spec["language"] == "en" else "french"]
    rows = [{"system": name, "ndcg@10": scores[spec["column"]], "source": "publié"}
            for name, scores in published.items() if spec["column"] in scores]
    rows += [{"system": name, "ndcg@10": score, "source": "mesuré ici"} for name, score in ours.items()]
    return sorted(rows, key=lambda row: -row["ndcg@10"])


def evaluate(dataset: str, data_dir: Path, knowledge_dir: Path | None = None, output: Path | None = None, *,
             partial: bool = False) -> dict:
    """Mesurer BM25, dense et RRF au niveau des pages, puis le contexte que recevrait le modèle.

    Tous les PDF du jeu doivent être ingérés : sur un corpus plus petit, le score ne serait plus comparable
    à ceux publiés. `partial=True` ne sert qu'à un essai de la chaîne sur quelques documents.
    """
    from docling_hybrid_rag import load_knowledge_base
    from docling_hybrid_rag.benchmarks import environment
    from docling_hybrid_rag.page_evaluation import VARIANTS, evaluate_queries
    from docling_hybrid_rag.retrieval import encode_question

    target = dataset_dir(data_dir, dataset)
    knowledge = load_knowledge_base(knowledge_dir or target / "knowledge")
    ingested = {record["title"] for record in knowledge.records.values()}
    missing = set(load_documents(target)) - ingested
    if missing and not partial:
        raise ValueError(f"{len(missing)} PDF ne sont pas ingérés (étape ingest) : le score ne serait pas comparable.")
    queries = load_queries(target, dataset)
    if partial:
        for query in queries:
            query["relevance"] = {page: note for page, note in query["relevance"].items() if page[0] in ingested}
        queries = [query for query in queries if query["relevance"]]
    vectors = {query["query_id"]: encode_question(query["question"], knowledge) for query in queries}
    measured = evaluate_queries(knowledge, queries, vectors)
    spec = DATASETS[dataset]
    results = {
        "environment": environment(dataset=f"vidore/vidore_v3_{dataset}", dataset_revision=spec["revision"],
                                   paper=PAPER),
        "dataset": {"name": dataset, "title": spec["title"], "language": spec["language"], "questions": len(queries),
                    "documents": len(ingested),
                    "pages": len(_parquet(next((target / "corpus").glob("test-*.parquet")), ["corpus_id"])),
                    "children": len(knowledge.children), "parents": len(knowledge.parents)},
        "partial": partial,
        "variants": {variant: summarize(measured["variants"][variant]) for variant in VARIANTS},
        "context": summarize(measured["context"]),
    }
    by_type: dict[str, dict] = {}
    for kind in sorted({t for q in queries for t in q["query_types"]}):
        ids = {q["query_id"] for q in queries if kind in q["query_types"]}
        by_type[kind] = summarize([r for r in measured["variants"]["rrf"] if r["query_id"] in ids])
    results["rrf_by_query_type"] = by_type
    # Vérification du protocole : le BM25 sur le texte fourni doit retrouver le BM25S publié.
    published = PUBLISHED["english" if spec["language"] == "en" else "french"]["BM25S (texte)"][spec["column"]]
    results["protocol_check"] = {**baseline_bm25(dataset, data_dir), "published_bm25s_ndcg@10": published}
    ours = {f"docling-hybrid-rag, {name}": 100 * results["variants"][name]["ndcg@10"]["mean"] for name in VARIANTS}
    results["comparison"] = comparison(dataset, ours)
    if output:
        Path(output).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["download", "baseline", "ingest", "evaluate"])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="hr")
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks/vidore-v3"))
    parser.add_argument("--knowledge-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="documents à ingérer (essai court)")
    args = parser.parse_args()
    if args.stage == "download":
        print(download(args.dataset, args.data_dir))
    elif args.stage == "baseline":
        print(json.dumps(baseline_bm25(args.dataset, args.data_dir)))
    elif args.stage == "ingest":
        ingest(args.dataset, args.data_dir, args.knowledge_dir, only=args.only)
    else:
        output = dataset_dir(args.data_dir, args.dataset) / "results.json"
        results = evaluate(args.dataset, args.data_dir, args.knowledge_dir, output)
        for row in results["comparison"]:
            print(f"{row['ndcg@10']:6.1f}  {row['system']}  ({row['source']})")


if __name__ == "__main__":
    main()
