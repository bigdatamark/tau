# [MODULE DOC]
# Tests for tau_coding.image_processing: content-based MIME detection for JPEG/PNG/GIF/WEBP/BMP,
# rejection of JPEG XL, animated PNG and malformed headers with explicit reasons, BMP-to-PNG
# conversion, aspect-preserving downscaling to 2000px, no upscaling/re-encoding of small images,
# encoded-size bounded downscaling, source byte/pixel limits, and safe failure objects for
# encode/decode errors. Uses Pillow to synthesize images in memory.
from io import BytesIO

import pytest
from PIL import Image

import tau_coding.image_processing as image_processing
from tau_coding.image_processing import (
    ImageProcessingFailure,
    ProcessedImage,
    detect_supported_image_mime_type,
    process_image,
    unsupported_image_reason,
)


def image_bytes(format_name: str, *, size: tuple[int, int] = (16, 12)) -> bytes:
    # [FUNCTION DOC]
    # HELPER: Build an in-memory solid teal RGB image of the given size (default 16x12) encoded in
    # the named Pillow format and return its bytes.
    output = BytesIO()
    Image.new("RGB", size, "teal").save(output, format=format_name)
    return output.getvalue()


@pytest.mark.parametrize(
    ("format_name", "mime_type"),
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("GIF", "image/gif"),
        ("WEBP", "image/webp"),
        ("BMP", "image/bmp"),
    ],
)
def test_detects_supported_images_by_content(format_name: str, mime_type: str) -> None:
    # [FUNCTION DOC]
    # Parametrized test: each supported format's bytes are detected as the expected MIME type.
    assert detect_supported_image_mime_type(image_bytes(format_name)) == mime_type


def test_rejects_jpeg_xl_and_malformed_headers() -> None:
    # [FUNCTION DOC]
    # Test: JPEG XL signature, a PNG signature followed by garbage, and a bare BMP header are all
    # not detected as supported (None).
    assert detect_supported_image_mime_type(b"\xff\xd8\xff\xf7not-jpeg") is None
    assert detect_supported_image_mime_type(b"\x89PNG\r\n\x1a\nnot-a-png") is None
    assert detect_supported_image_mime_type(b"BM" + b"\x00" * 40) is None


def test_rejects_animated_png_before_image_data() -> None:
    # [FUNCTION DOC]
    # Test: inserting an acTL chunk before IDAT makes a PNG animated; detection returns None and the
    # unsupported reason says animated PNG images are not supported.
    png = image_bytes("PNG")
    idat_offset = png.index(b"IDAT") - 4
    animated_chunk = b"\x00\x00\x00\x08acTL\x00\x00\x00\x02\x00\x00\x00\x00" + b"\x00" * 4
    animated = png[:idat_offset] + animated_chunk + png[idat_offset:]

    assert detect_supported_image_mime_type(animated) is None
    assert unsupported_image_reason(animated) == "animated PNG images are not supported"


def test_jpeg_xl_has_explicit_unsupported_reason() -> None:
    # [FUNCTION DOC]
    # Test: JPEG XL bytes are not detected and get the reason "JPEG XL images are not supported".
    data = b"\xff\xd8\xff\xf7not-jpeg"

    assert detect_supported_image_mime_type(data) is None
    assert unsupported_image_reason(data) == "JPEG XL images are not supported"


def test_bmp_is_converted_to_png() -> None:
    # [FUNCTION DOC]
    # Test: BMP input is converted to PNG output with a PNG signature and a conversion note.
    data = image_bytes("BMP")

    result = process_image(data, "image/bmp")

    assert isinstance(result, ProcessedImage)
    assert result.mime_type == "image/png"
    assert result.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert "Image converted from image/bmp to image/png." in result.notes


def test_large_dimensions_are_resized_without_changing_aspect_ratio() -> None:
    # [FUNCTION DOC]
    # Test: a 2500x1000 PNG is resized to 2000x800 and a resize note is recorded.
    data = image_bytes("PNG", size=(2_500, 1_000))

    result = process_image(data, "image/png")

    assert isinstance(result, ProcessedImage)
    assert (result.width, result.height) == (2_000, 800)
    assert "Image resized from 2500x1000 to 2000x800." in result.notes


def test_small_images_are_not_upscaled_or_reencoded() -> None:
    # [FUNCTION DOC]
    # Test: a small PNG keeps its dimensions and original bytes, with no notes.
    data = image_bytes("PNG", size=(20, 10))

    result = process_image(data, "image/png")

    assert isinstance(result, ProcessedImage)
    assert (result.width, result.height) == (20, 10)
    assert result.data == data
    assert result.notes == ()


def test_encoded_size_limit_triggers_bounded_downscaling() -> None:
    # [FUNCTION DOC]
    # Test: a noisy 300x300 PNG with a 3000-byte max is downscaled until encoded size fits.
    image = Image.effect_noise((300, 300), 100).convert("RGB")
    output = BytesIO()
    image.save(output, format="PNG")

    result = process_image(output.getvalue(), "image/png", max_bytes=3_000)

    assert isinstance(result, ProcessedImage)
    assert len(result.data) <= 3_000
    assert result.width < 300
    assert result.height < 300


def test_source_byte_limit_omits_image_before_decoding() -> None:
    # [FUNCTION DOC]
    # Test: source bytes exceeding max_source_bytes yield an ImageProcessingFailure mentioning the
    # processing limit, without decoding.
    data = image_bytes("PNG")

    result = process_image(data, "image/png", max_source_bytes=len(data) - 1)

    assert isinstance(result, ImageProcessingFailure)
    assert "processing limit" in result.message


def test_source_pixel_limit_omits_image_before_loading_pixels() -> None:
    # [FUNCTION DOC]
    # Test: a 4x4 image (16 pixels) over a 15-pixel max_source_pixels yields a failure message
    # naming both counts.
    data = image_bytes("PNG", size=(4, 4))

    result = process_image(data, "image/png", max_source_pixels=15)

    assert isinstance(result, ImageProcessingFailure)
    assert "16 pixels" in result.message
    assert "15-pixel processing limit" in result.message


def test_encode_failure_returns_safe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test: monkeypatches the private encoder to raise OSError; processing returns a failure whose
    # message is "could not encode processed image (encoder unavailable)".
    def fail_encode(image: Image.Image, mime_type: str, attempt: int) -> bytes:
        # [FUNCTION DOC]
        # HELPER: Replacement encoder that ignores its arguments and raises OSError("encoder unavailable").
        del image, mime_type, attempt
        raise OSError("encoder unavailable")

    monkeypatch.setattr(image_processing, "_encode_image", fail_encode)

    result = process_image(image_bytes("BMP"), "image/bmp")

    assert isinstance(result, ImageProcessingFailure)
    assert result.message == "could not encode processed image (encoder unavailable)"


def test_decode_failure_returns_safe_failure() -> None:
    # [FUNCTION DOC]
    # Test: a PNG with a valid signature/IHDR tag but zeroed contents is detected as PNG yet processing
    # returns a failure saying it could not decode a valid image.
    malformed = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\x0dIHDR" + b"\x00" * 17 + b"\x00\x00\x00\x00IDAT" + b"\x00" * 4
    )
    assert detect_supported_image_mime_type(malformed) == "image/png"

    result = process_image(malformed, "image/png")

    assert isinstance(result, ImageProcessingFailure)
    assert "could not decode a valid image" in result.message
