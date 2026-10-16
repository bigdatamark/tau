# [MODULE DOC]
# Tests that each provider payload builder in tau_ai carries image content from
# tool results to the provider in that provider's native shape (Anthropic,
# OpenAI Responses, Codex, OpenAI Chat, Google Gemini, Mistral), and that
# non-vision models get a text placeholder instead of image bytes.
# Calls private payload-builder functions directly; no network I/O.
from tau_agent import ImageContent, TextContent, ToolResultMessage
from tau_ai.anthropic import _build_messages_payload
from tau_ai.google import _build_google_payload
from tau_ai.mistral import _build_mistral_payload
from tau_ai.openai_codex import _build_codex_payload
from tau_ai.openai_compatible import _build_chat_payload, _build_responses_payload


def _image_result() -> ToolResultMessage:
    # [FUNCTION DOC]
    # HELPER: fixture-like builder of a tool result message for tool "read" (call
    # id "call-1") containing a text block and a base64 PNG image block.
    return ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[
            TextContent(text="Read image file [image/png]"),
            ImageContent(data="aW1hZ2U=", mime_type="image/png"),
        ],
    )


def test_anthropic_embeds_image_in_tool_result() -> None:
    # [FUNCTION DOC]
    # Asserts the Anthropic messages payload nests the image as the second
    # content item of the tool_result block, using a base64 source with media type
    # and data.
    payload = _build_messages_payload(
        model="claude",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    result = payload["messages"][0]["content"][0]
    assert result["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2U=",
        },
    }


def test_openai_responses_embeds_image_in_function_output() -> None:
    # [FUNCTION DOC]
    # Asserts the OpenAI Responses payload's first input item (function output)
    # has an input_image second element with detail "auto" and a data URL.
    payload = _build_responses_payload(
        model="gpt",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    result = payload["input"][0]
    assert result["output"][1] == {
        "type": "input_image",
        "detail": "auto",
        "image_url": "data:image/png;base64,aW1hZ2U=",
    }


def test_codex_embeds_image_in_function_output() -> None:
    # [FUNCTION DOC]
    # Asserts the Codex payload's function output second element is an
    # input_image.
    payload = _build_codex_payload(
        model="codex",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    result = payload["input"][0]
    assert result["output"][1]["type"] == "input_image"


def test_openai_chat_attaches_tool_images_in_followup_user_message() -> None:
    # [FUNCTION DOC]
    # Asserts the Chat Completions payload emits system, tool, then a follow-up
    # user message whose second content part is an image_url (tool messages cannot
    # hold images there).
    payload = _build_chat_payload(
        model="vision",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    assert [message["role"] for message in payload["messages"]] == ["system", "tool", "user"]
    assert payload["messages"][2]["content"][1]["type"] == "image_url"


def test_google_embeds_image_in_gemini_3_function_response() -> None:
    # [FUNCTION DOC]
    # Asserts that for a Gemini 3 model the functionResponse part embeds the image
    # as inlineData with mime type and base64 data.
    payload = _build_google_payload(
        model="gemini-3-pro",
        system="system",
        messages=[_image_result()],
        tools=[],
        reasoning_effort=None,
        max_tokens=None,
        supports_images=True,
    )

    response = payload["contents"][0]["parts"][0]["functionResponse"]
    assert response["parts"][0] == {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}


def test_mistral_attaches_tool_images_in_followup_user_message() -> None:
    # [FUNCTION DOC]
    # Asserts the Mistral payload uses system, tool, user messages with the image
    # attached as an image_url part of the follow-up user message.
    payload = _build_mistral_payload(
        model="pixtral",
        system="system",
        messages=[_image_result()],
        tools=[],
        reasoning_effort=None,
        max_tokens=None,
        supports_images=True,
    )

    assert [message["role"] for message in payload["messages"]] == ["system", "tool", "user"]
    assert payload["messages"][2]["content"][1]["type"] == "image_url"


def test_non_vision_models_receive_placeholder_instead_of_image() -> None:
    # [FUNCTION DOC]
    # Asserts that with images unsupported the Responses function output contains
    # the "image contents are unavailable" and "switch to a vision-capable model"
    # placeholder text and that the base64 data appears nowhere in the payload.
    payload = _build_responses_payload(
        model="text-only",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=False,
    )

    result = payload["input"][0]
    assert "image contents are unavailable—do not infer or describe them" in result["output"]
    assert "switch to a vision-capable model" in result["output"]
    assert "aW1hZ2U=" not in str(payload)
