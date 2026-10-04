"""Parents et enfants construits avec les primitives Docling de l'article.

`HybridChunker` produit les enfants dans la limite du tokenizer. `TreeChunkExpander`
remonte chaque enfant vers les éléments documentaires complets qui le contiennent :
ces éléments forment son parent.
"""

from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.chunk_expander import TreeChunkExpander
from docling_core.types.doc import DocItem, DoclingDocument
from langchain_core.documents import Document

PARENT_ID_KEY = "parent_id"


def _item_label(item: DocItem) -> str:
    label = item.label
    return label.value if hasattr(label, "value") else str(label)


def _pages(items: Sequence[DocItem]) -> list[int]:
    return sorted({provenance.page_no for item in items for provenance in item.prov})


def _build_parents(document: DoclingDocument, tokenizer, document_path: str | None):
    chunker = HybridChunker(
        tokenizer=tokenizer,
        # Regrouper les unités voisines qui partagent le même contexte Docling
        # réduit les parents minuscules sans modifier la limite des enfants.
        merge_peers=True,
        repeat_table_header=False,
    )
    serializer = chunker.serializer_provider.get_serializer(doc=document)
    expander = TreeChunkExpander()
    item_by_ref = {item.self_ref: item for item, _ in document.iterate_items()}
    parent_id_by_item_ref: dict[str, str] = {}
    parent_records: OrderedDict[tuple[str, ...], Document] = OrderedDict()
    chunks_with_parent = []

    for chunk in chunker.chunk(dl_doc=document):
        if not chunk.meta.doc_items:
            continue
        expanded = expander.expand(chunk, document, serializer)
        refs = tuple(item.self_ref for item in expanded.meta.doc_items)
        if not refs:
            continue
        if refs not in parent_records:
            parent_id = f"parent-{len(parent_records) + 1:03d}"
            headings = list(chunk.meta.headings or [])
            pages = _pages(expanded.meta.doc_items)
            parent_records[refs] = Document(page_content=expanded.text, metadata={
                PARENT_ID_KEY: parent_id,
                "document_path": document_path,
                "document_title": document.name,
                "heading_path": " > ".join(headings),
                "heading_refs": [item.self_ref for item in document.texts if item.text in headings],
                "doc_item_refs": list(refs),
                "picture_refs": [],
                "page_start": min(pages, default=None),
                "page_end": max(pages, default=None),
                "raw_text": expanded.text,
                "contextualized_text": chunker.contextualize(expanded),
                "token_count": tokenizer.count_tokens(expanded.text),
            })
            for item in expanded.meta.doc_items:
                parent_id_by_item_ref[item.self_ref] = parent_id
        chunks_with_parent.append((chunk, parent_records[refs].metadata[PARENT_ID_KEY]))

    parents = list(parent_records.values())
    parent_by_id = {parent.metadata[PARENT_ID_KEY]: parent for parent in parents}

    # Les images ne sont pas des chunks textuels. On les rattache au parent
    # le plus proche dans l'ordre Docling, sans inventer de description.
    body_refs = [ref.cref for ref in document.body.children]
    for body_index, ref in enumerate(body_refs):
        item = item_by_ref.get(ref)
        if item is None or _item_label(item) != "picture" or ref in parent_id_by_item_ref:
            continue
        before = (r for r in reversed(body_refs[:body_index]) if r in parent_id_by_item_ref)
        after = (r for r in body_refs[body_index + 1:] if r in parent_id_by_item_ref)
        nearest = next(before, None) or next(after, None)
        if nearest is not None:
            parent_by_id[parent_id_by_item_ref[nearest]].metadata["picture_refs"].append(ref)

    return chunker, parents, chunks_with_parent


def _build_children(chunker, parents, chunks_with_parent, tokenizer):
    children: list[Document] = []
    cursors: dict[str, int] = defaultdict(int)
    parent_text = {parent.metadata[PARENT_ID_KEY]: parent.metadata["raw_text"] for parent in parents}
    document_path = parents[0].metadata["document_path"] if parents else None

    for chunk, parent_id in chunks_with_parent:
        text = parent_text[parent_id]
        located_start = text.find(chunk.text, cursors[parent_id])
        if located_start < 0:
            located_start = cursors[parent_id]
        # Inclure les séparateurs entre deux objets dans l'enfant courant,
        # afin que les offsets forment une partition exacte du parent.
        start = cursors[parent_id]
        end = located_start + len(chunk.text)
        indexed_text = chunker.contextualize(chunk)
        children.append(Document(page_content=indexed_text, metadata={
            "child_id": f"child-{len(children) + 1:04d}",
            PARENT_ID_KEY: parent_id,
            "document_path": document_path,
            "doc_item_refs": [item.self_ref for item in chunk.meta.doc_items],
            "heading_path": " > ".join(chunk.meta.headings or []),
            "start": start, "end": end,
            "raw_text": text[start:end],
            "token_count": tokenizer.count_tokens(indexed_text),
        }))
        cursors[parent_id] = end

    # Les séparateurs éventuels du sérialiseur restent récupérables comme texte brut.
    for parent in parents:
        parent_id = parent.metadata[PARENT_ID_KEY]
        text, cursor = parent_text[parent_id], cursors[parent_id]
        if cursor < len(text):
            children.append(Document(page_content=text[cursor:], metadata={
                "child_id": f"child-{len(children) + 1:04d}",
                PARENT_ID_KEY: parent_id,
                "document_path": parent.metadata["document_path"],
                "doc_item_refs": parent.metadata["doc_item_refs"],
                "heading_path": parent.metadata["heading_path"],
                "start": cursor, "end": len(text),
                "raw_text": text[cursor:],
                "token_count": tokenizer.count_tokens(text[cursor:]),
                "synthetic_tail": True,
            }))
    return children


def build_parent_child_chunks(document: DoclingDocument, tokenizer,
                              document_path: str | Path | None = None):
    """Construire les parents restitués au modèle et les enfants indexés.

    Le texte d'un enfant (`page_content`) ajoute les titres et légendes de son
    contexte Docling : c'est lui que BM25 et le dense indexent. `raw_text` et les
    offsets `start`/`end` découpent exactement le texte du parent.
    """
    path = str(document_path) if document_path is not None else None
    chunker, parents, chunks_with_parent = _build_parents(document, tokenizer, path)
    return parents, _build_children(chunker, parents, chunks_with_parent, tokenizer)
