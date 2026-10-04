"""Pipeline Docling de l'article : configurer, parser, sauvegarder, recharger."""

from importlib.metadata import version
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions, HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode

from docling_hybrid_rag.store import code_fingerprint, file_sha256, record_stage, require_stage

PARSING = "parsing"
# Réglages de vitesse ou de mémoire : ils ne changent pas le document produit,
# donc ils n'invalident pas un parsing déjà enregistré.
PERFORMANCE_OPTIONS = {
    "accelerator_options", "layout_batch_size", "table_batch_size", "ocr_batch_size",
    "queue_max_size", "batch_polling_interval_seconds", "document_timeout", "artifacts_path",
}


def default_pdf_options() -> PdfPipelineOptions:
    """Les options présentées dans l'article."""
    return PdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,
        do_formula_enrichment=True,
        code_formula_options=CodeFormulaVlmOptions.from_preset(
            "granite_docling", scale=1.0, max_size=1_024
        ),
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        generate_parsed_pages=True,
    )


def build_document_converter(pdf_options: PdfPipelineOptions | None = None) -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options or default_pdf_options()),
        },
    )


def parse_document(
    source: str | Path,
    document_converter: DocumentConverter,
) -> DoclingDocument:
    """Convertit le PDF et refuse un résultat partiel."""

    result = document_converter.convert(
        source,
        raises_on_error=False,
        max_num_pages=1_000,
        max_file_size=512 * 1024 * 1024,
    )

    if result.status != ConversionStatus.SUCCESS:
        messages = "; ".join(error.error_message for error in result.errors)
        raise RuntimeError(
            f"Échec du parsing ({result.status.value}) : {messages}"
        )

    return result.document


def parsing_inputs(document_dir: Path, pdf_options: PdfPipelineOptions | None = None) -> dict:
    """Ce qui détermine le document parsé : le PDF, les options, Docling et ce code."""
    options = (pdf_options or default_pdf_options()).model_dump(mode="json", exclude=PERFORMANCE_OPTIONS)
    return {
        "source_sha256": file_sha256(Path(document_dir) / "source.pdf"),
        "options": options,
        "versions": {name: version(name) for name in ("docling", "docling-core")},
        "code": code_fingerprint(*_PRODUCERS),
    }


def save_parsed_document(
    doc: DoclingDocument,
    document_dir: Path,
    pdf_options: PdfPipelineOptions | None = None,
) -> Path:
    """Enregistre le document structuré et ses images, puis le manifeste du parsing."""

    document_dir = Path(document_dir).resolve()
    canonical_path = document_dir / "document.json"

    # Le JSON conserve structure, provenance, annotations et images référencées.
    doc.save_as_json(
        canonical_path,
        # Chemin relatif au dossier du JSON pour conserver des références portables.
        artifacts_dir=Path("artifacts"),
        image_mode=ImageRefMode.REFERENCED,
    )
    record_stage(document_dir, PARSING, parsing_inputs(document_dir, pdf_options), ["document.json"])
    return canonical_path


def load_document(canonical_path: Path) -> DoclingDocument:
    doc = DoclingDocument.load_from_json(canonical_path)
    for picture in doc.pictures:
        if picture.image is not None:
            # Le JSON garde des chemins relatifs à son dossier ; Docling a besoin du chemin complet.
            picture.image.uri = canonical_path.parent / Path(picture.image.uri)
            if not Path(picture.image.uri).is_file():
                raise FileNotFoundError(f"Image manquante : {picture.image.uri}")
    return doc


def load_parsed_document(document_dir: Path, pdf_options: PdfPipelineOptions | None = None) -> DoclingDocument:
    """Recharger le document sans reparser, si le manifeste correspond à ces options."""
    document_dir = Path(document_dir).resolve()
    require_stage(document_dir, PARSING, parsing_inputs(document_dir, pdf_options))
    return load_document(document_dir / "document.json")


# Fonctions dont le code détermine le résultat du parsing. Elles sont capturées
# ici pour que l'empreinte reste celle du vrai code, même si un test les remplace.
_PRODUCERS = (parse_document, save_parsed_document)
