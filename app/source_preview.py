"""Render a source PDF page and mark the region behind a comparison result."""
from __future__ import annotations

from io import BytesIO

import pymupdf
from PIL import Image, ImageDraw


def _region(page, text: str | None, bbox: tuple[float, float, float, float] | None):
    fallback = pymupdf.Rect(bbox) if bbox else None
    if text and len(text) <= 400:
        words = text.split()
        queries = [(text, "Exact text match")]
        if len(words) > 10:
            queries.extend(((" ".join(words[:16]), "Matched text fragments"),
                            (" ".join(words[-16:]), "Matched text fragments")))
        fragment_hits = []
        for query, quality in queries:
            hits = page.search_for(query)
            if fallback:
                vicinity = pymupdf.Rect(fallback.x0 - 8, fallback.y0 - 8,
                                         fallback.x1 + 8, fallback.y1 + 8)
                hits = [hit for hit in hits if hit.intersects(vicinity)]
            else:
                hits = hits[:1]
            if hits:
                if quality == "Exact text match":
                    region = pymupdf.Rect(hits[0])
                    for hit in hits[1:]:
                        region |= hit
                    return region, quality
                fragment_hits.extend(hits)
        if fragment_hits:
            region = pymupdf.Rect(fragment_hits[0])
            for hit in fragment_hits[1:]:
                region |= hit
            return region, "Matched text fragments"
    if fallback and not fallback.is_empty:
        return fallback, "Extracted source region"
    return None, "Page available; exact region unavailable"


def render_source_page(pdf: bytes, page_number: int, text: str | None,
                       bbox: tuple[float, float, float, float] | None,
                       scale: float = 1.7) -> tuple[bytes, bytes | None, str]:
    """Return (marked full page PNG, focused crop PNG, location quality)."""
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Page {page_number} is outside this PDF")
        page = doc[page_number - 1]
        page.set_rotation(0)  # extraction rectangles use unrotated page coordinates
        region, label = _region(page, text, bbox)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGBA")
        crop = None
        if region and region.intersects(page.rect):
            region &= page.rect
            draw = ImageDraw.Draw(image, "RGBA")
            box = (region.x0 * scale, region.y0 * scale,
                   region.x1 * scale, region.y1 * scale)
            draw.rectangle(box, fill=(210, 102, 58, 45), outline=(173, 63, 24, 255), width=4)
            margin = 70 * scale
            focus = (max(0, int(box[0] - margin)), max(0, int(box[1] - margin)),
                     min(image.width, int(box[2] + margin)),
                     min(image.height, int(box[3] + margin)))
            focused = image.crop(focus)
            stream = BytesIO()
            focused.convert("RGB").save(stream, format="PNG")
            crop = stream.getvalue()
        stream = BytesIO()
        image.convert("RGB").save(stream, format="PNG")
        return stream.getvalue(), crop, label
    finally:
        doc.close()
