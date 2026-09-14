"""Mêmes sections, blocs et offsets que dans les cellules parents et enfants."""
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from langchain_core.documents import Document
from docling.chunking import HierarchicalChunker
from docling_core.types.doc import DocItem, PictureItem, TitleItem, SectionHeaderItem
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from lib.settings import CHILD_MAX_TOKENS

PARENT_ID_KEY = "parent_id"

def build_parent_chunks(doc, canonical_path: Path, tokenizer):
    hierarchical_chunks = list(HierarchicalChunker().chunk(doc))
    parents = []
    parent_parts = {}
    serializer = MarkdownDocSerializer(doc=doc, params=MarkdownParams(image_placeholder=""))

    # Les figures sans texte ne sont pas émises par le chunker textuel.
    # Nous retrouvons leur section à partir des titres et niveaux du document.
    headings_by_level = {}
    section_by_item_ref = {}
    items_by_section = defaultdict(list)
    for item, _ in doc.iterate_items():                                           # <2>
        if isinstance(item, (TitleItem, SectionHeaderItem)):
            level = item.level if isinstance(item, SectionHeaderItem) else 0
            headings_by_level = {
                key: value for key, value in headings_by_level.items() if key < level
            }
            headings_by_level[level] = item.self_ref
        elif isinstance(item, DocItem):
            section_key = tuple(headings_by_level.values())
            section_by_item_ref[item.self_ref] = section_key
            items_by_section[section_key].append(item)                            # <2>

    for parent_number, (section_key, section) in enumerate(
        groupby(
            hierarchical_chunks,
            key=lambda chunk: section_by_item_ref[chunk.meta.doc_items[0].self_ref],
        ),
        start=1,
    ):
        section_chunks = list(section)
        headings = section_chunks[0].meta.headings or []
        parent_id = f"parent-{parent_number:03d}"
        doc_items = items_by_section[section_key]
        # Le parent garde les tableaux en Markdown et les formules transcrites.
        # Les figures sont jointes séparément comme images dans le contexte.
        parts = [{"text": "\n\n".join(
            f'{"#" * depth} {heading}' for depth, heading in enumerate(headings, 1)
        ), "refs": list(section_key)}] if headings else []
        visited = set()
        for item in doc_items:
            if item.self_ref in visited or item.self_ref in serializer.get_excluded_refs():
                continue
            result = serializer.serialize(item=item, visited=visited)
            if result.text.strip():
                parts.append({"text": result.text.strip(), "refs": [item.self_ref]})
        parent_parts[parent_id] = parts
        parent_text = "\n\n".join(part["text"] for part in parts)
        pages = sorted({prov.page_no for item in doc_items for prov in item.prov})

        parents.append(Document(
            page_content=parent_text,
            metadata={
                PARENT_ID_KEY: parent_id,
                "document_path": str(canonical_path),
                "document_title": doc.name,
                "heading_path": " > ".join(headings),
                "heading_refs": list(section_key),
                "doc_item_refs": [item.self_ref for item in doc_items],
                "picture_refs": [
                    item.self_ref for item in doc_items if isinstance(item, PictureItem)
                ],
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                "token_count": tokenizer.count_tokens(parent_text),
            },
        ))

    return parents, parent_parts


def build_child_chunks(parents, parent_parts, canonical_path: Path, tokenizer):
    children = []
    for parent in parents:
        parent_id = parent.metadata[PARENT_ID_KEY]
        heading = parent.metadata["heading_path"]
        groups = [{"text": "", "refs": []}]
        for part_number, part in enumerate(parent_parts[parent_id]):              # <1>
            text = ("\n\n" if part_number else "") + part["text"]
            candidate = groups[-1]["text"] + text
            indexed_candidate = heading + "\n\n" + candidate
            if groups[-1]["text"] and tokenizer.count_tokens(indexed_candidate) > CHILD_MAX_TOKENS:
                groups.append({"text": "", "refs": []})
            groups[-1]["text"] += text
            groups[-1]["refs"].extend(part["refs"])                                # <1>

        start = 0
        for group in groups:
            raw_text = group["text"]
            end = start + len(raw_text)                                          # <2>
            indexed_text = heading + "\n\n" + raw_text if heading else raw_text
            children.append(Document(
                page_content=indexed_text,
                metadata={
                    "child_id": f"child-{len(children) + 1:03d}",
                    PARENT_ID_KEY: parent_id,                                    # <3>
                    "document_path": str(canonical_path),
                    "doc_item_refs": list(dict.fromkeys(group["refs"])),
                    "heading_path": heading,
                    "start": start, "end": end,
                    "raw_text": raw_text,
                    "token_count": tokenizer.count_tokens(indexed_text),
                },
            ))
            start = end
    return children
