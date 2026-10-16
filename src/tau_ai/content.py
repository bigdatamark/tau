# [MODULE DOC]
# HELPER: Shared helpers used by provider serializers to handle multimodal (text + image)
# message content. Defines two placeholder strings that replace images when the active model
# lacks vision support: one for user messages and one for tool results (the latter also tells
# the model to ask the user to switch to a vision-capable model). No I/O or state.
"""Shared helpers for provider serialization of multimodal message content."""

from __future__ import annotations

from collections.abc import Sequence

from tau_agent.messages import ImageContent, TextContent, ToolResultMessage, UserMessage

NON_VISION_USER_IMAGE_PLACEHOLDER = (
    "(image omitted: current model does not support image input; image contents are "
    "unavailable—do not infer or describe them)"
)
NON_VISION_TOOL_IMAGE_PLACEHOLDER = (
    "(tool image omitted: current model does not support image input; image contents are "
    "unavailable—do not infer or describe them; ask the user to switch to a vision-capable model)"
)


def messages_have_images(messages: Sequence[object]) -> bool:
    # [FUNCTION DOC]
    # HELPER: Detect whether a conversation context contains any image blocks.
    # Input: a sequence of arbitrary message objects.
    # Output: True if any user message or tool-result message has list-form content containing
    # at least one image content block; string content and other message types are ignored.
    # No side effects.
    """Return whether user or tool-result context contains image blocks."""
    return any(
        isinstance(message, (UserMessage, ToolResultMessage))
        and not isinstance(message.content, str)
        and any(isinstance(block, ImageContent) for block in message.content)
        for message in messages
    )


def text_and_images(
    content: str | Sequence[TextContent | ImageContent],
    *,
    supports_images: bool,
    image_placeholder: str,
) -> tuple[str, list[ImageContent]]:
    # [FUNCTION DOC]
    # HELPER: Split message content into visible text and sendable images.
    # Inputs: content (plain string, or sequence of text/image blocks); keyword-only flag whether
    # the model supports images; the placeholder text to use when images must be dropped.
    # Behavior: a string is returned unchanged with no images. For block lists, all text blocks are
    # concatenated with no separator and image blocks collected. If images are supported, returns
    # (text, images). Otherwise images are dropped; if any existed, the placeholder is appended on a
    # new line after the text (or becomes the whole text when text is empty).
    # Output: a pair of (text string, list of image blocks). No exceptions or side effects.
    """Return visible text and sendable images, downgrading unsupported images."""
    if isinstance(content, str):
        return content, []

    text = "".join(block.text for block in content if isinstance(block, TextContent))
    images = [block for block in content if isinstance(block, ImageContent)]
    if supports_images:
        return text, images
    if images:
        text = f"{text}\n{image_placeholder}" if text else image_placeholder
    return text, []
