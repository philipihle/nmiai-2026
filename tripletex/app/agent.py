import base64
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from .models import FileInput, TripletexCredentials
from .prompts.system_prompt import SYSTEM_PROMPT  # kept as fallback for load_prompt_from_gcs
from .improver import load_prompt_from_gcs
from .tools.http_client import TripletexClient
from .tools import registry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30
MODEL = "gemini-2.5-pro"


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
    system_prompt = load_prompt_from_gcs()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    tripletex = TripletexClient(credentials.base_url, credentials.session_token)
    gemini_tools = _build_gemini_tools(registry.get_all_tools())

    # Add a done tool so the model can signal completion explicitly
    done_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="task_complete",
            description="Call this when you have finished all necessary API calls and the task is complete.",
            parameters={"type": "object", "properties": {"summary": {"type": "string"}}},
        )
    ])
    all_tools = gemini_tools + [done_tool]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=all_tools,
        temperature=0.1,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=_build_initial_parts(prompt, files))
    ]

    # Loop detection: track (tool_name, args_str) call counts
    tool_call_counts: dict[str, int] = {}

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"Agent iteration {iteration + 1}/{MAX_ITERATIONS}")

        for _attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as _e:
                if _attempt == 2:
                    raise
                logger.warning(f"Gemini call failed (attempt {_attempt+1}): {_e}, retrying...")
                import asyncio as _asyncio
                await _asyncio.sleep(2 ** _attempt)

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
        done = False
        result_parts = []
        for part in fn_calls:
            fn = part.function_call
            # Deep-convert protobuf MapComposite/RepeatedComposite to plain Python types
            args = json.loads(json.dumps(dict(fn.args) if fn.args else {}))

            if fn.name == "task_complete":
                logger.info("Agent called task_complete — finished")
                done = True
                result_parts.append(types.Part.from_function_response(
                    name=fn.name,
                    response={"result": {"status": "ok"}},
                ))
                continue

            logger.info(f"Tool call: {fn.name}({json.dumps(args, ensure_ascii=False, default=str)[:200]})")

            # Loop detection: same tool+args called too many times → force done
            call_key = f"{fn.name}:{json.dumps(args, sort_keys=True, default=str)[:150]}"
            tool_call_counts[call_key] = tool_call_counts.get(call_key, 0) + 1
            if tool_call_counts[call_key] > 3:
                logger.warning(f"Loop detected: {fn.name} called {tool_call_counts[call_key]} times with same args — forcing task_complete")
                result_parts.append(types.Part.from_function_response(
                    name=fn.name,
                    response={"result": {"error": "Loop detected: this tool was already called with the same arguments. Stop retrying and call task_complete."}},
                ))
                done = True
                continue

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

        if done:
            break

    else:
        logger.warning(f"Agent reached max iterations ({MAX_ITERATIONS})")
