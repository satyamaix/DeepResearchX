"""
PDF Extraction Tool for DRX Deep Research System.

Provides text and metadata extraction from PDF documents.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from src.utils.url_validator import SSRFError, validate_url

logger = logging.getLogger(__name__)


class PDFMetadata(TypedDict):
    """Metadata extracted from a PDF document."""
    title: str | None
    author: str | None
    subject: str | None
    creator: str | None
    creation_date: str | None
    modification_date: str | None
    page_count: int


class PageContent(TypedDict):
    """Content from a single PDF page."""
    page_number: int
    text: str
    char_count: int


class Table(TypedDict):
    """A table extracted from PDF (placeholder for future enhancement)."""
    page_number: int
    data: list[list[str]]
    headers: list[str] | None


class ExtractedDocument(TypedDict):
    """Complete extraction result from a PDF."""
    text: str
    pages: list[PageContent]
    tables: list[Table]
    metadata: PDFMetadata
    extraction_method: str
    source: str
    extracted_at: str


class PDFExtractor:
    """
    Extract text and metadata from PDF documents.

    Supports extraction from:
    - Local file paths
    - URLs (downloads first)
    - Raw bytes
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_pages: int = 100,
        user_agent: str = "DRX-Research/1.0",
    ):
        self._timeout = timeout
        self._max_pages = max_pages
        self._user_agent = user_agent

    async def extract(self, source: str | bytes | Path) -> ExtractedDocument:
        """
        Extract text and metadata from a PDF source.

        Args:
            source: File path, URL, or raw PDF bytes

        Returns:
            ExtractedDocument with full extraction results
        """
        pdf_bytes: bytes
        source_str: str

        if isinstance(source, bytes):
            pdf_bytes = source
            source_str = "bytes"
        elif isinstance(source, Path):
            pdf_bytes = source.read_bytes()
            source_str = str(source)
        elif source.startswith(("http://", "https://")):
            pdf_bytes = await self._download_pdf(source)
            source_str = source
        else:
            # Assume file path
            pdf_bytes = Path(source).read_bytes()
            source_str = source

        return self._extract_from_bytes(pdf_bytes, source_str)

    async def _download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        # Validate URL to prevent SSRF attacks
        try:
            validate_url(url)
        except SSRFError as e:
            raise ValueError(f"Invalid URL for PDF extraction: {e}")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                url,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                logger.warning(f"URL may not be PDF: {content_type}")

            return response.content

    def _extract_from_bytes(self, pdf_bytes: bytes, source: str) -> ExtractedDocument:
        """Extract content from PDF bytes."""
        reader = PdfReader(io.BytesIO(pdf_bytes))

        pages: list[PageContent] = []
        all_text_parts: list[str] = []

        page_count = min(len(reader.pages), self._max_pages)

        for i in range(page_count):
            page = reader.pages[i]
            text = page.extract_text() or ""

            pages.append(PageContent(
                page_number=i + 1,
                text=text,
                char_count=len(text),
            ))
            all_text_parts.append(text)

        metadata = self._extract_metadata(reader)

        return ExtractedDocument(
            text="\n\n".join(all_text_parts),
            pages=pages,
            tables=[],  # Table extraction is a future enhancement
            metadata=metadata,
            extraction_method="pypdf",
            source=source,
            extracted_at=datetime.utcnow().isoformat() + "Z",
        )

    def _extract_metadata(self, reader: PdfReader) -> PDFMetadata:
        """Extract metadata from PDF."""
        meta = reader.metadata or {}

        def safe_date(val: Any) -> str | None:
            if val is None:
                return None
            try:
                return str(val)
            except Exception:
                return None

        return PDFMetadata(
            title=meta.get("/Title"),
            author=meta.get("/Author"),
            subject=meta.get("/Subject"),
            creator=meta.get("/Creator"),
            creation_date=safe_date(meta.get("/CreationDate")),
            modification_date=safe_date(meta.get("/ModDate")),
            page_count=len(reader.pages),
        )

    def get_metadata_sync(self, source: str | bytes | Path) -> PDFMetadata:
        """Synchronously get just the metadata (no full extraction)."""
        if isinstance(source, bytes):
            pdf_bytes = source
        elif isinstance(source, Path):
            pdf_bytes = source.read_bytes()
        else:
            pdf_bytes = Path(source).read_bytes()

        reader = PdfReader(io.BytesIO(pdf_bytes))
        return self._extract_metadata(reader)


# Factory function
def create_pdf_extractor(**kwargs: Any) -> PDFExtractor:
    """Create a configured PDFExtractor instance."""
    return PDFExtractor(**kwargs)


__all__ = [
    "PDFExtractor",
    "ExtractedDocument",
    "PDFMetadata",
    "PageContent",
    "Table",
    "create_pdf_extractor",
]
