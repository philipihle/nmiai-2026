import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import run_agent
from .improver import run_improvement
from .models import SolveRequest, SolveResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

AGENT_TIMEOUT = 285  # seconds — 15s buffer before competition's 300s hard limit


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/improve")
async def improve_endpoint():
    asyncio.create_task(run_improvement())
    return {"status": "started"}


@app.post("/solve")
async def solve(request: SolveRequest) -> SolveResponse:
    logger.info(f"Received task: {request.prompt[:120]!r}")
    try:
        await asyncio.wait_for(
            run_agent(request.prompt, request.files, request.tripletex_credentials),
            timeout=AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Agent timed out — returning completed anyway")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)

    return SolveResponse(status="completed")
