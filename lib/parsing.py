"""Pipeline Docling repris de l'article, puis rechargement de ses images."""
from pathlib import Path
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions, HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode

def build_document_converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,                                         # <1>
        do_formula_enrichment=True,                                           # <2>
        code_formula_options=CodeFormulaVlmOptions.from_preset(
            "granite_docling", scale=1.0, max_size=1_024
        ),
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),       # <3>
        generate_parsed_pages=True,                                            # <3>
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
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

    if result.status != ConversionStatus.SUCCESS:                              # <1>
        messages = "; ".join(error.error_message for error in result.errors)
        raise RuntimeError(
            f"Échec du parsing ({result.status.value}) : {messages}"
        )

    return result.document


def save_parsed_document(
    doc: DoclingDocument,
    output_dir: Path,
) -> Path:
    """Enregistre le document structuré et ses images."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "document.json"

    # Le JSON conserve structure, provenance, annotations et images référencées.
    doc.save_as_json(
        canonical_path,
        # Chemin relatif au dossier du JSON pour conserver des références portables.
        artifacts_dir=Path("artifacts"),
        image_mode=ImageRefMode.REFERENCED,
    )

    return canonical_path


def load_document(canonical_path: Path) -> DoclingDocument:
    doc = DoclingDocument.load_from_json(canonical_path)
    for picture in doc.pictures:
        if picture.image is not None:
            picture.image.uri = canonical_path.parent / Path(picture.image.uri)
    return doc
