"""Parents et enfants construits avec les primitives Docling de l'article.

Les enfants sont les chunks de `HybridChunker`, bornés par le tokenizer : petits, pour
une recherche précise. Le parent est le texte complet des éléments du document que ses
enfants touchent, sérialisé par `TreeChunkExpander` : un tableau découpé en plusieurs
chunks revient entier. Par défaut, tous les enfants d'une même section partagent le
même parent : large et complet, pour le contexte donné au modèle.

Deux enfants qui touchent un même élément du document ont toujours le même parent. Sans
cette règle, un tableau découpé en quatre chunks donnait quatre parents différents, dont
certains contenaient les autres. Les parents forment ainsi une partition du texte,
et les offsets des enfants partitionnent exactement leur parent.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.chunk_expander import TreeChunkExpander
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItem, DoclingDocument
from langchain_core.documents import Document

PARENT_ID_KEY = "parent_id"
# Taille maximale d'un parent par section. Un contexte de sept parents reste sous 15 000 tokens.
MAX_PARENT_TOKENS = 2048


class MarkdownTableProvider(ChunkingSerializerProvider):
    """Sérialiser les tableaux en Markdown, comme dans l'exemple de la documentation de Docling.

    Le format par défaut répète la ligne et la colonne devant chaque cellule
    (« N-3, Période de développement.1 = 27 »). Le Markdown garde la forme du tableau,
    que le modèle lit mieux, et il est plus court.
    """

    def get_serializer(self, doc):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


def _item_label(item: DocItem) -> str:
    label = item.label
    return label.value if hasattr(label, "value") else str(label)


def _pages(items: Sequence[DocItem]) -> list[int]:
    return sorted({provenance.page_no for item in items for provenance in item.prov})


def _merge_groups(links: list[set[str]]) -> list[list[int]]:
    """Réunir les chunks qui partagent un élément, puis les renvoyer dans l'ordre du document."""
    parent = list(range(len(links)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner: dict = {}
    for index, refs in enumerate(links):
        for key in refs:
            if key in owner:
                parent[find(index)] = find(owner[key])
            owner.setdefault(key, index)
    groups = defaultdict(list)
    for index in range(len(links)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=min)


def _pack_sections(groups: list[list[int]], keys: list, sizes: list[int], limit: int | None) -> list[list[int]]:
    """Réunir des groupes consécutifs d'une même section, sans dépasser `limit` tokens.

    `sizes` donne la taille du texte de chaque groupe, élargissement compris. Un groupe (par exemple
    un tableau ou une liste de références découpés en plusieurs chunks) n'est jamais coupé : il peut
    dépasser la limite à lui seul. Une section plus longue donne plusieurs parents consécutifs.
    """
    packed, last_key, total = [], None, 0
    for group, size in zip(groups, sizes):
        key = keys[group[0]]
        if packed and key is not None and key == last_key and (limit is None or total + size <= limit):
            packed[-1].extend(group)
            total += size
        else:
            packed.append(list(group))
            last_key, total = key, size
    return packed


def _end_of_match(text: str, piece: str, start: int) -> int:
    """Position où finit `piece` dans `text` à partir de `start`, sans tenir compte des espaces.

    Le texte d'un chunk sépare ses éléments par une ligne vide, celui d'un parent par un
    simple retour à la ligne : seul le contenu doit coïncider.
    """
    words = piece.split()
    match = re.compile(r"\s+".join(map(re.escape, words))).search(text, start)
    if match is None:
        raise ValueError(f"Un chunk est introuvable dans son parent : {piece[:120]!r}")
    return match.end()


def _children_ends(text: str, pieces: list[str]) -> list[int]:
    """Fin de chaque enfant dans le parent. Les enfants se suivent dans l'ordre du document.

    Le dernier va jusqu'au bout du parent, séparateurs compris : aucun enfant ne contient
    seulement des séparateurs, et les offsets partitionnent le texte.
    """
    cursor, ends = 0, []
    for piece in pieces:
        cursor = _end_of_match(text, piece, cursor)
        ends.append(cursor)
    ends[-1] = len(text)
    return ends


def _section_keys(chunks) -> list:
    """Clé de section de chaque chunk : son chemin de titres et le numéro de sa série consécutive.

    Deux chunks ne partagent une section que s'ils se suivent sous le même titre : un titre
    répété plus loin (« Résultats ») n'en fait pas un seul parent. Un chunk sans titre n'a pas
    de clé, sinon tout le texte d'avant le premier titre deviendrait un seul parent.
    """
    keys, previous, run = [], None, 0
    for chunk in chunks:
        headings = tuple(chunk.meta.headings or ())
        if headings != previous:
            run, previous = run + 1, headings
        keys.append((headings, run) if headings else None)
    return keys


def build_parent_child_chunks(document: DoclingDocument, tokenizer, document_path: str | Path | None = None, *,
                              parent_scope: str = "section", markdown_tables: bool = True,
                              max_parent_tokens: int | None = MAX_PARENT_TOKENS):
    """Construire les parents restitués au modèle et les enfants indexés.

    Les enfants sont petits, pour une recherche précise. Les parents sont larges et complets,
    pour que le modèle lise le contexte entier.

    `parent_scope="section"` (par défaut) réunit tous les enfants d'une même section : le parent
    garde les hypothèses, les formules, les tableaux et les figures voisins du passage retrouvé.
    Une section de plus de `max_parent_tokens` donne plusieurs parents consécutifs, car un parent
    de plusieurs milliers de tokens noie le passage utile.
    `parent_scope="elements"` donne à chaque parent seulement les éléments touchés par ses
    enfants : il est souvent identique à son enfant unique.

    Le texte d'un enfant (`page_content`) ajoute les titres et légendes de son contexte
    Docling : c'est lui que BM25 et le dense indexent. `raw_text`, `start` et `end`
    découpent exactement le texte du parent.
    """
    if parent_scope not in ("elements", "section"):
        raise ValueError("parent_scope doit valoir 'elements' ou 'section'.")
    chunker = HybridChunker(
        tokenizer=tokenizer,
        # Regrouper les unités voisines qui partagent le même contexte Docling
        # réduit les parents minuscules sans modifier la limite des enfants.
        merge_peers=True,
        repeat_table_header=False,
        **({"serializer_provider": MarkdownTableProvider()} if markdown_tables else {}),
    )
    serializer = chunker.serializer_provider.get_serializer(doc=document)
    expander = TreeChunkExpander()
    chunks = [chunk for chunk in chunker.chunk(dl_doc=document) if chunk.meta.doc_items and chunk.text.strip()]
    if not chunks:
        return [], []

    expansions = [{item.self_ref for item in expander.expand(chunk, document, serializer).meta.doc_items}
                  for chunk in chunks]
    def expand_members(members):
        items = list({item.self_ref: item for index in members for item in chunks[index].meta.doc_items}.values())
        merged = DocChunk(text="", meta=DocMeta(doc_items=items, headings=chunks[members[0]].meta.headings,
                                                origin=document.origin))
        return items, expander.expand(merged, document, serializer)

    groups = _merge_groups(expansions)
    if parent_scope == "section":
        # La taille d'un groupe est celle de son texte élargi : une liste de références compte en entier.
        sizes = [tokenizer.count_tokens(expand_members(group)[1].text) for group in groups]
        groups = _pack_sections(groups, _section_keys(chunks), sizes, max_parent_tokens)

    path = str(document_path) if document_path is not None else None
    item_by_ref = {item.self_ref: item for item, _ in document.iterate_items()}
    parents, children, parent_id_by_item_ref = [], [], {}
    for number, members in enumerate(groups, start=1):
        items, expanded = expand_members(members)
        first = chunks[members[0]]
        # Chaque élément se termine par un retour à la ligne : le parent s'arrête au dernier mot.
        text, parent_items, origin = expanded.text.strip(), expanded.meta.doc_items, "tree"
        try:
            ends = _children_ends(text, [chunks[index].text for index in members])
        except ValueError:
            # Dans un document dont les paragraphes sont rangés sous leur titre (Markdown, DOCX),
            # l'élargissement ne renvoie que le titre et perd le texte des chunks. Le parent est alors
            # la réunion de ses enfants : il ne contient jamais moins qu'eux, et la partition reste exacte.
            text = "\n\n".join(chunks[index].text.strip() for index in members)
            parent_items, origin = items, "chunks"
            ends = _children_ends(text, [chunks[index].text for index in members])
        parent_id = f"parent-{number:03d}"
        headings = list(first.meta.headings or [])
        pages = _pages(parent_items)
        parents.append(Document(page_content=text, metadata={
            PARENT_ID_KEY: parent_id,
            "document_path": path,
            "document_title": document.name,
            "heading_path": " > ".join(headings),
            "heading_refs": [item.self_ref for item in document.texts if item.text in headings],
            "doc_item_refs": [item.self_ref for item in parent_items],
            "picture_refs": [],
            "page_start": min(pages, default=None),
            "page_end": max(pages, default=None),
            "raw_text": text,
            "contextualized_text": chunker.contextualize(DocChunk(text=text, meta=expanded.meta)),
            "token_count": tokenizer.count_tokens(text),
            # « tree » : texte complet des éléments touchés ; « chunks » : réunion des enfants.
            "parent_origin": origin,
        }))
        for item in parent_items:
            parent_id_by_item_ref[item.self_ref] = parent_id

        # Chaque enfant commence où le précédent finit.
        start = 0
        for index, end in zip(members, ends):
            indexed_text = chunker.contextualize(chunks[index])
            children.append(Document(page_content=indexed_text, metadata={
                "child_id": f"child-{len(children) + 1:04d}",
                PARENT_ID_KEY: parent_id,
                "document_path": path,
                "doc_item_refs": [item.self_ref for item in chunks[index].meta.doc_items],
                "heading_path": " > ".join(chunks[index].meta.headings or []),
                "start": start, "end": end,
                "raw_text": text[start:end],
                "token_count": tokenizer.count_tokens(indexed_text),
            }))
            start = end

    # Les images ne sont pas des chunks textuels. On les rattache au parent
    # le plus proche dans l'ordre Docling, sans inventer de description.
    parent_by_id = {parent.metadata[PARENT_ID_KEY]: parent for parent in parents}
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
    return parents, children
