"""Open RAG Benchmark (Vectara, articles arXiv) : une base de plusieurs PDF et la mesure de la recherche.

Les questions, les sections attendues et les PDF viennent de https://huggingface.co/datasets/vectara/open_ragbench
(licence CC BY-NC 4.0, usage non commercial). Ils sont téléchargés en local et ne sont jamais publiés dans ce dépôt.

Étapes, à lancer l'une après l'autre (un modèle lourd à la fois sur un PC de 8 Go) :

    python benchmarks/open_rag_bench.py download   # questions, sections attendues, PDF (pause entre deux requêtes arXiv)
    python benchmarks/open_rag_bench.py ingest     # parse, découpe et indexe chaque PDF
    python benchmarks/open_rag_bench.py evaluate   # BM25, dense, RRF, reranker facultatif

Chaque étape reprend où elle s'est arrêtée. `--data-dir` reçoit les fichiers du benchmark, `--knowledge-dir` la base.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
# Dix articles de 8 à 24 pages, dix questions chacun, avec des tableaux et des figures dans leurs questions.
DOCUMENTS = ["2410.14077v2", "2407.18337v4", "2408.02322v2", "2410.08642v2", "2410.08147v8",
             "2405.05998v3", "2412.18252v2", "2409.02603v3", "2409.13674v3", "2410.11074v3"]
USER_AGENT = "docling-hybrid-rag-benchmark"


def fetch(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def download(data_dir: Path, documents: list[str]) -> None:
    for name in ("queries", "qrels", "answers", "pdf_urls"):
        fetch(f"{HF}/{name}.json", data_dir / f"{name}.json")
    urls = json.loads((data_dir / "pdf_urls.json").read_text(encoding="utf-8"))
    for document in documents:
        fetch(urls[document], data_dir / "pdfs" / f"{document}.pdf")
        time.sleep(4)  # arXiv demande de ne pas enchaîner les requêtes.
        section_file = data_dir / "sections" / f"{document}.json"
        if not section_file.exists():
            corpus = data_dir / "corpus" / f"{document}.json"
            fetch(f"{HF}/corpus/{document}.json", corpus)
            # Les images en base64 ne servent pas à la mesure : on ne garde que le texte et les tableaux.
            sections = [{"text": s["text"], "tables": s.get("tables", {})}
                        for s in json.loads(corpus.read_text(encoding="utf-8"))["sections"]]
            section_file.parent.mkdir(parents=True, exist_ok=True)
            section_file.write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")
            corpus.unlink()


def light_options():
    """Parsing sans transcription des formules ni images de figures : la mesure porte sur le texte et les tableaux."""
    from docling_hybrid_rag.parsing import default_pdf_options

    options = default_pdf_options()
    options.do_formula_enrichment = False
    options.generate_picture_images = False
    return options


def ingest(data_dir: Path, knowledge_dir: Path, documents: list[str], scope: str) -> None:
    from docling_hybrid_rag import ingest_document

    for document in documents:
        started = time.perf_counter()
        record = ingest_document(data_dir / "pdfs" / f"{document}.pdf", knowledge_dir, title=document,
                                 pdf_options=light_options(), parent_scope=scope)
        print(f"{document}: {record['parents']} parents, {record['children']} enfants, "
              f"{time.perf_counter() - started:.0f} s", flush=True)


def build_cases(data_dir: Path, documents: list[str]) -> list[dict]:
    from docling_hybrid_rag.passage_evaluation import shingles

    queries = json.loads((data_dir / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((data_dir / "qrels.json").read_text(encoding="utf-8"))
    sections = {d: json.loads((data_dir / "sections" / f"{d}.json").read_text(encoding="utf-8")) for d in documents}
    cases = []
    for query_id, relevance in qrels.items():
        document = relevance["doc_id"]
        if document not in sections:
            continue
        section = sections[document][relevance["section_id"]]
        # Les questions sur un tableau sont attendues dans la section et ses tableaux.
        gold_text = " ".join([section["text"], *section["tables"].values()])
        cases.append({"query_id": query_id, "question": queries[query_id]["query"], "document": document,
                      "source": queries[query_id]["source"], "type": queries[query_id]["type"],
                      "gold": shingles(gold_text)})
    return cases


def print_summary(label: str, variant: str, summary: dict) -> None:
    low, high = summary["hit@3"]["low"], summary["hit@3"]["high"]
    print(f"[{label}] {variant:13s} n={summary['questions']:3d} "
          f"doc_hit@3 {summary['doc_hit@3']['mean']:.2f} | hit@1 {summary['hit@1']['mean']:.2f} "
          f"@3 {summary['hit@3']['mean']:.2f} [{low:.2f}-{high:.2f}] @5 {summary['hit@5']['mean']:.2f} | "
          f"MRR {summary['reciprocal_rank']['mean']:.2f} | "
          f"couverture@3 {summary['coverage@3']['mean']:.2f} | tokens@3 {summary['tokens@3']['mean']:.0f}", flush=True)


def evaluate(data_dir: Path, knowledge_dir: Path, documents: list[str], scopes: list[str],
             reranker_questions: int, output: Path) -> None:
    from docling_hybrid_rag import load_knowledge_base
    from docling_hybrid_rag.passage_evaluation import (
        evaluate_ranking, overlaps, rank_all_variants, shingles, summarize, with_parent_scope,
    )
    from docling_hybrid_rag.retrieval import encode_question
    from docling_hybrid_rag.settings import get_reranker

    knowledge = load_knowledge_base(knowledge_dir)
    cases = build_cases(data_dir, documents)
    child_shingles = [(c.metadata["document_title"], shingles(c.metadata["raw_text"])) for c in knowledge.children]
    # Une question dont la section n'apparaît dans aucun enfant (parsing différent) ne mesure pas la recherche.
    alignable = [c for c in cases if any(t == c["document"] and overlaps(c["gold"], s) for t, s in child_shingles)]
    print(f"{len(cases)} questions, {len(alignable)} dont la section est retrouvable dans le parsing", flush=True)

    vectors = {c["query_id"]: encode_question(c["question"], knowledge) for c in alignable}
    reranker = get_reranker() if reranker_questions else None
    results = {"questions": len(cases), "alignable": len(alignable), "scopes": {}}
    for scope in scopes:
        base = knowledge if scope == "section" else with_parent_scope(knowledge, knowledge_dir, scope)
        parents = base.parents
        rows = {variant: [] for variant in ("bm25", "dense", "rrf", "rrf+reranker")}
        for number, case in enumerate(alignable):
            use_reranker = reranker if number < reranker_questions else None
            rankings = rank_all_variants(case["question"], vectors[case["query_id"]], base, use_reranker)
            for variant, ranking in rankings.items():
                row = evaluate_ranking(base, ranking, case["gold"], case["document"])
                rows[variant].append({"query_id": case["query_id"], "source": case["source"], "type": case["type"], **row})
        results["scopes"][scope] = {"parents": len(parents), "variants": {
            variant: {"all": summarize(items), **{
                source: summarize([r for r in items if r["source"] == source])
                for source in sorted({r["source"] for r in items})}}
            for variant, items in rows.items() if items}}
        for variant, items in rows.items():
            if items:
                print_summary(f"{scope}, {len(parents)} parents", variant, summarize(items))
        if reranker_questions:
            # Le reranker ne passe que sur les premières questions : on compare les autres moteurs sur les mêmes.
            paired = {variant: summarize(items[:reranker_questions]) for variant, items in rows.items() if items}
            results["scopes"][scope]["paired"] = paired
            for variant, summary in paired.items():
                print_summary(f"{scope}, mêmes {reranker_questions} questions", variant, summary)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"résultats dans {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["download", "ingest", "evaluate"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks/open-rag-bench-arxiv"))
    parser.add_argument("--knowledge-dir", type=Path, default=Path("data/benchmarks/open-rag-bench-arxiv/knowledge"))
    parser.add_argument("--documents", nargs="*", default=DOCUMENTS)
    parser.add_argument("--parent-scope", choices=["section", "elements"], default="section")
    parser.add_argument("--scopes", nargs="*", default=["section", "elements"], help="portées comparées par `evaluate`")
    parser.add_argument("--reranker-questions", type=int, default=0, help="nombre de questions passées au reranker")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.stage == "download":
        download(args.data_dir, args.documents)
    elif args.stage == "ingest":
        ingest(args.data_dir, args.knowledge_dir, args.documents, args.parent_scope)
    else:
        output = args.output or args.data_dir / "results.json"
        evaluate(args.data_dir, args.knowledge_dir, args.documents, args.scopes, args.reranker_questions, output)


if __name__ == "__main__":
    main()
