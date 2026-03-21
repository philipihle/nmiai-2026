import base64
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from .models import FileInput, TripletexCredentials
from .prompts.system_prompt import SYSTEM_PROMPT
from .tools.http_client import TripletexClient
from .tools import registry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 25
MODEL = "gemini-2.0-flash-001"


def _build_gemini_tools(tools: list[dict]) -> list[types.Tool]:
    """Convert Anthropic-style tool dicts to Gemini FunctionDeclarations."""
    declarations = []
    for t in tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=schema,
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _build_initial_parts(prompt: str, files: list[FileInput]) -> list[types.Part]:
    parts: list[types.Part] = [types.Part.from_text(text=prompt)]

    for f in files:
        raw = base64.b64decode(f.content_base64)
        if f.mime_type == "application/pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=raw, filetype="pdf")
                parts.append(types.Part.from_text(text=f"[Attached file: {f.filename}]"))
                for page in doc:
                    pix = page.get_pixmap(dpi=150)
                    parts.append(types.Part.from_bytes(
                        data=pix.tobytes("png"),
                        mime_type="image/png",
                    ))
            except Exception as e:
                logger.warning(f"PDF conversion failed for {f.filename}: {e}")
                parts.append(types.Part.from_text(text=f"[File {f.filename} could not be rendered]"))
        elif f.mime_type in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            parts.append(types.Part.from_text(text=f"[Attached file: {f.filename}]"))
            parts.append(types.Part.from_bytes(data=raw, mime_type=f.mime_type))
        else:
            try:
                parts.append(types.Part.from_text(text=f"[File: {f.filename}]\n{raw.decode('utf-8')}"))
            except Exception:
                parts.append(types.Part.from_text(text=f"[Binary file: {f.filename}, type: {f.mime_type}]"))

    return parts


async def run_agent(
    prompt: str,
    files: list[FileInput],
    credentials: TripletexCredentials,
) -> None:
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GCP_PROJECT", "ai-nm26osl-1847"),
        location=os.environ.get("GCP_LOCATION", "europe-west1"),
    )
    tripletex = TripletexClient(credentials.base_url, credentials.session_token)
    gemini_tools = _build_gemini_tools(registry.get_all_tools())

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=gemini_tools,
        temperature=0.1,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=_build_initial_parts(prompt, files))
    ]

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"Agent iteration {iteration + 1}/{MAX_ITERATIONS}")

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        assistant_content = candidate.content
        contents.append(assistant_content)

        logger.info(f"Finish reason: {candidate.finish_reason}")

        # Check for function calls in the response
        fn_calls = [
            p for p in (assistant_content.parts or [])
            if p.function_call is not None
        ]

        if not fn_calls:
            logger.info("No function calls — agent finished")
            break

        # Execute all function calls and collect results
        result_parts = []
        for part in fn_calls:
            fn = part.function_call
            args = dict(fn.args) if fn.args else {}
            logger.info(f"Tool call: {fn.name}({json.dumps(args, ensure_ascii=False, default=str)[:200]})")

            try:
                result = await registry.dispatch(fn.name, args, tripletex)
            except Exception as e:
                logger.error(f"Tool {fn.name} raised: {e}")
                result = {"error": str(e)}

            # Truncate large list responses
            if isinstance(result, dict) and isinstance(result.get("body"), dict):
                values = result["body"].get("values", [])
                if len(values) > 20:
                    result["body"]["values"] = values[:20]
                    result["body"]["_truncated"] = True

            logger.info(f"Tool result: {json.dumps(result, default=str)[:300]}")

            result_parts.append(types.Part.from_function_response(
                name=fn.name,
                response={"result": result},
            ))

        contents.append(types.Content(role="user", parts=result_parts))

    else:
        logger.warning(f"Agent reached max iterations ({MAX_ITERATIONS})")
