"""Restituer exactement le texte des parents, leurs images et leurs références."""

import base64
import json
import mimetypes
import re
from pathlib import Path
from docling_core.types.doc import RefItem

from lib.parsing import load_document


def parent_content_blocks(parent):
    # Le visuel et le modèle reçoivent le même texte et les mêmes octets d'image.
    blocks = [{"type": "text", "text": parent["text"]}]
    for image in parent["images"]:
        path = Path(image["path"])
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{encoded}", "detail": "high",
        }})
    return blocks


def source_id(document_id, ref):
    # Deux PDF peuvent tous deux avoir une figure #/pictures/0.
    return f"{document_id}:{ref}"


def build_sources(context, documents):
    sources = {}
    for parent in context:
        doc = documents[parent["metadata"]["document_id"]]
        meta = parent["metadata"]
        section = meta["heading_path"] or "Début du document"
        common = {"parent_id": parent["parent_id"], "document_title": meta["document_title"],
                  "source_url": parent["source_url"]}
        items = [RefItem(cref=ref).resolve(doc) for ref in meta["doc_item_refs"]]
        sources[parent["parent_id"]] = {
            **common, "name": section, "pages": sorted({p.page_no for item in items for p in item.prov}),
            "text": parent["text"], "ref": None,
        }
        for item in items:
            kind = item.label.value
            if kind not in ("picture", "table", "formula"):
                continue
            caption = " ".join(c.resolve(doc).text for c in getattr(item, "captions", [])).strip()
            name = {"picture": "Figure", "table": "Tableau", "formula": "Formule"}[kind]
            text = getattr(item, "text", caption)
            number = re.search(r"\\tag\{([^}]+)\}", text)
            name += f" {number.group(1)}" if kind == "formula" and number else ""
            name += f" « {caption} »" if caption else " (sans titre conservé)"
            excerpt = item.export_to_markdown(doc=doc) if kind == "table" else text
            sources[source_id(meta["document_id"], item.self_ref)] = {
                **common, "name": f"{section}, {name}", "pages": sorted({p.page_no for p in item.prov}),
                "text": excerpt, "ref": item.self_ref,
            }
    return sources


def build_context(parent_ids, knowledge_base):
    """Rassembler les parents retenus. Aucun recadrage ni texte inventé n'est ajouté."""
    context, documents = [], {}
    for parent_id in dict.fromkeys(parent_ids):
        parent = knowledge_base.parents_by_id[parent_id]
        meta = parent.metadata
        document_id = meta["document_id"]
        path = knowledge_base.data_dir / meta["document_path"]
        if document_id not in documents:
            documents[document_id] = load_document(path)
        doc = documents[document_id]
        pictures = {item.self_ref: item for item in doc.pictures}
        images = []
        for ref in meta["picture_refs"]:
            picture = pictures[ref]
            if picture.image is None or not Path(picture.image.uri).is_file():
                raise FileNotFoundError(f"Image manquante pour {document_id}:{ref}.")
            images.append({"picture_ref": source_id(document_id, ref), "path": str(picture.image.uri)})
        record = knowledge_base.records[document_id]
        source_url = record["source_url"] or path.with_name("source.pdf").as_uri()
        context.append({"parent_id": parent_id, "text": parent.page_content,
                        "metadata": meta, "images": images, "source_url": source_url})
    return {"parents": context, "sources": build_sources(context, documents)}
