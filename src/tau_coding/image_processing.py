# [MODULE DOC]
# Image detection and normalization for image attachments read by coding tools
# (tau_coding layer). Detects image types from magic bytes, rejects unsupported
# variants (JPEG XL, animated PNG), and decodes/re-encodes images with Pillow under
# strict limits: 50MB source size, 40 million source pixels, 2000px max dimension
# and 5MB encoded output by default, with up to 12 encode/shrink attempts.
# Decompression-bomb warnings are escalated to errors. All functions are pure apart
# from CPU/memory use; failures are returned as values, not raised.
"""Bounded image detection and normalization for coding-tool attachments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError

DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_SOURCE_IMAGE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_IMAGE_DIMENSION = 2_000
DEFAULT_MAX_SOURCE_PIXELS = 40_000_000
MAX_RESIZE_ATTEMPTS = 12
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
type PngKind = Literal["static", "animated", "invalid"]


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    # [CLASS DOC]
    # Immutable result of successful processing: encoded bytes, MIME type, final width
    # and height, and a tuple of human-readable notes about conversions/resizes applied
    # (empty when the original bytes were passed through).
    """Provider-safe encoded image and notes about transformations applied."""

    data: bytes
    mime_type: str
    width: int
    height: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageProcessingFailure:
    # [CLASS DOC]
    # Immutable value carrying a safe, user-facing message explaining why an image was
    # not attached.
    """Safe user-facing reason why an image attachment was omitted."""

    message: str


def detect_image_family_mime_type(data: bytes) -> str | None:
    # [FUNCTION DOC]
    # Classify bytes into a broad image family by magic number only, without checking
    # validity: JPEG XL (checked before JPEG since it shares the prefix), JPEG, PNG,
    # GIF, WebP (RIFF with WEBP tag), BMP. Returns the MIME type string or None.
    """Identify a known image family from the minimum available magic bytes."""
    if data.startswith(b"\xff\xd8\xff\xf7"):
        return "image/jxl"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def unsupported_image_reason(data: bytes) -> str | None:
    # [FUNCTION DOC]
    # Return a user-facing reason for recognized-but-unsupported images: JPEG XL, or a
    # PNG whose chunk scan finds an animation control chunk before image data.
    # Returns None for everything else (including unrecognized data).
    """Explain recognized image variants that Tau intentionally cannot attach."""
    if data.startswith(b"\xff\xd8\xff\xf7"):
        return "JPEG XL images are not supported"
    if data.startswith(PNG_SIGNATURE) and _classify_png(data) == "animated":
        return "animated PNG images are not supported"
    return None


def detect_supported_image_mime_type(data: bytes) -> str | None:
    # [FUNCTION DOC]
    # Return the MIME type only for images Tau will attach: JPEG (not JPEG XL), static
    # PNG with a well-formed chunk structure, GIF, WebP with a VP8/VP8L/VP8X chunk, and
    # BMP with a plausible header. Returns None otherwise.
    """Detect a supported image from its bytes while rejecting unsafe variants."""
    if data.startswith(b"\xff\xd8\xff"):
        return None if data.startswith(b"\xff\xd8\xff\xf7") else "image/jpeg"
    if data.startswith(PNG_SIGNATURE):
        return "image/png" if _classify_png(data) == "static" else None
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        data.startswith(b"RIFF")
        and len(data) >= 16
        and data[8:12] == b"WEBP"
        and data[12:16] in {b"VP8 ", b"VP8L", b"VP8X"}
    ):
        return "image/webp"
    if data.startswith(b"BM") and _is_valid_bmp_header(data):
        return "image/bmp"
    return None


def process_image(
    data: bytes,
    mime_type: str,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_IMAGE_BYTES,
    max_source_pixels: int = DEFAULT_MAX_SOURCE_PIXELS,
) -> ProcessedImage | ImageProcessingFailure:
    # [FUNCTION DOC]
    # Validate an image and normalize it to provider-safe limits.
    # Inputs: raw bytes, declared MIME type, keyword limits (max output bytes, max
    # dimension, max source bytes, max source pixels).
    # Steps: reject sources over the source byte limit; read size/animation flag and
    # verify the image (any decode/bomb error becomes a failure); reject if pixel count
    # exceeds the limit; pass through unchanged when the type is not BMP and it is
    # within dimension and byte limits. Otherwise reject animated images; fully decode
    # pixels; BMP is converted to PNG (with a note); downscale preserving aspect ratio to
    # fit the max dimension; then repeatedly encode (JPEG/WebP quality decreasing by 8
    # per attempt down to 45, GIF/PNG optimized) and if still too large shrink by 20%
    # each attempt, until it fits or attempts run out or size stops changing.
    # Output: ProcessedImage (with a resize note when dimensions changed) or
    # ImageProcessingFailure. Raises nothing intentionally; encode errors become failures.
    """Validate and, when needed, normalize an image within deterministic limits."""
    if len(data) > max_source_bytes:
        return ImageProcessingFailure(
            f"source is {_format_size(len(data))}, exceeding the "
            f"{_format_size(max_source_bytes)} processing limit"
        )

    try:
        width, height, animated = _validated_image_metadata(data)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        return ImageProcessingFailure(f"could not decode a valid image ({error})")

    if width * height > max_source_pixels:
        return ImageProcessingFailure(
            f"image has {width * height:,} pixels, exceeding the {max_source_pixels:,}-pixel "
            "processing limit"
        )

    requires_conversion = mime_type == "image/bmp"
    requires_resize = width > max_dimension or height > max_dimension or len(data) > max_bytes
    if not requires_conversion and not requires_resize:
        return ProcessedImage(data=data, mime_type=mime_type, width=width, height=height)

    if animated:
        return ImageProcessingFailure(
            "animated image exceeds inline limits and cannot be resized without losing animation"
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as opened:
                opened.load()
                image = opened.copy()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as error:
        return ImageProcessingFailure(f"could not decode image pixels ({error})")

    output_mime = "image/png" if requires_conversion else mime_type
    notes: list[str] = []
    if output_mime != mime_type:
        notes.append(f"Image converted from {mime_type} to {output_mime}.")

    target_width, target_height = _bounded_dimensions(width, height, max_dimension)
    if (target_width, target_height) != (width, height):
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

    for attempt in range(MAX_RESIZE_ATTEMPTS):
        try:
            encoded = _encode_image(image, output_mime, attempt)
        except Exception as error:
            return ImageProcessingFailure(f"could not encode processed image ({error})")
        if len(encoded) <= max_bytes:
            final_width, final_height = image.size
            if (final_width, final_height) != (width, height):
                notes.append(
                    f"Image resized from {width}x{height} to {final_width}x{final_height}."
                )
            return ProcessedImage(
                data=encoded,
                mime_type=output_mime,
                width=final_width,
                height=final_height,
                notes=tuple(notes),
            )

        next_width = max(1, int(image.width * 0.8))
        next_height = max(1, int(image.height * 0.8))
        if (next_width, next_height) == image.size:
            break
        image = image.resize((next_width, next_height), Image.Resampling.LANCZOS)

    return ImageProcessingFailure(
        f"could not resize image below the {_format_size(max_bytes)} attachment limit"
    )


def _validated_image_metadata(data: bytes) -> tuple[int, int, bool]:
    # [FUNCTION DOC]
    # HELPER: Open bytes with Pillow (bomb warnings as errors), read width, height and
    # animated flag, and run Pillow's verify. Returns a triple of width, height, animated.
    # Raises Pillow/OS/value errors on invalid data, ValueError for non-positive size.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            animated = bool(getattr(image, "is_animated", False))
            image.verify()
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return width, height, animated


def _encode_image(image: Image.Image, mime_type: str, attempt: int) -> bytes:
    # [FUNCTION DOC]
    # HELPER: Encode a Pillow image to bytes in the given MIME format. JPEG converts to
    # RGB and uses quality max(45, 90 - 8*attempt) with optimization; WebP uses the same
    # quality ramp; GIF optimized; anything else is encoded as optimized PNG.
    # May raise Pillow errors (e.g. unsupported mode for a format).
    output = BytesIO()
    if mime_type == "image/jpeg":
        quality = max(45, 90 - attempt * 8)
        image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
    elif mime_type == "image/webp":
        quality = max(45, 90 - attempt * 8)
        image.save(output, format="WEBP", quality=quality, method=4)
    elif mime_type == "image/gif":
        image.save(output, format="GIF", optimize=True)
    else:
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _bounded_dimensions(width: int, height: int, maximum: int) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: Scale width and height down (never up) so neither exceeds the maximum,
    # preserving aspect ratio; each result is at least 1. Returns the new pair.
    scale = min(1.0, maximum / width, maximum / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _classify_png(data: bytes) -> PngKind:
    # [FUNCTION DOC]
    # HELPER: Walk PNG chunks after the signature. Returns "invalid" if the first chunk
    # is not a 13-byte IHDR, data is shorter than 33 bytes, or a chunk length overruns
    # the data; "animated" if an acTL chunk appears before the first IDAT; "static" when
    # IDAT is reached first; "invalid" if neither is found.
    if len(data) < 33 or int.from_bytes(data[8:12], "big") != 13 or data[12:16] != b"IHDR":
        return "invalid"
    offset = len(PNG_SIGNATURE)
    while offset + 12 <= len(data):
        chunk_length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type == b"acTL":
            return "animated"
        if chunk_type == b"IDAT":
            return "static"
        next_offset = offset + 12 + chunk_length
        if next_offset <= offset or next_offset > len(data):
            return "invalid"
        offset = next_offset
    return "invalid"


def _is_valid_bmp_header(data: bytes) -> bool:
    # [FUNCTION DOC]
    # HELPER: Sanity-check a BMP header: at least 30 bytes, declared file size (if
    # non-zero) at least 26, pixel data offset after the headers and before the declared
    # end, DIB header size 12 (core) or 40..124 (info variants), one color plane, and a
    # standard bit depth (1, 4, 8, 16, 24, 32). Returns a boolean.
    if len(data) < 30:
        return False
    declared_size = int.from_bytes(data[2:6], "little")
    pixel_offset = int.from_bytes(data[10:14], "little")
    dib_size = int.from_bytes(data[14:18], "little")
    if declared_size and declared_size < 26:
        return False
    if pixel_offset < 14 + dib_size or (declared_size and pixel_offset >= declared_size):
        return False
    if dib_size == 12:
        planes = int.from_bytes(data[22:24], "little")
        bits_per_pixel = int.from_bytes(data[24:26], "little")
    elif 40 <= dib_size <= 124:
        planes = int.from_bytes(data[26:28], "little")
        bits_per_pixel = int.from_bytes(data[28:30], "little")
    else:
        return False
    return planes == 1 and bits_per_pixel in {1, 4, 8, 16, 24, 32}


def _format_size(size: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Format a byte count as megabytes with one decimal, e.g. "5.0MB".
    return f"{size / (1024 * 1024):.1f}MB"
