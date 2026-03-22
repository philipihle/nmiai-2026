"""
Astar Island — Cloud Run endpoint
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from solver import AstarSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("astar")

app = FastAPI(title="Astar Island Solver")


def _get_token(body: dict) -> str:
    return body.get("token") or os.environ.get("ASTAR_TOKEN", "")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    """
    Main endpoint called by competition validators or manually.
    Accepts optional JSON body: {"token": "...", "round_id": "..."}
    Falls back to ASTAR_TOKEN env var for auth.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    token = _get_token(body)
    if not token:
        return JSONResponse(
            {"error": "No auth token. Set ASTAR_TOKEN env var or pass 'token' in body."},
            status_code=400,
        )

    round_id = body.get("round_id")
    use_mc = body.get("use_mc", True)
    solver = AstarSolver(token, use_mc=use_mc)

    try:
        result = solver.solve_round(round_id)
    except Exception as e:
        logger.exception("Solve failed")
        return JSONResponse({"error": str(e)}, status_code=500)

    return result


@app.get("/rounds")
async def rounds():
    """Utility: list all rounds."""
    token = os.environ.get("ASTAR_TOKEN", "")
    if not token:
        return JSONResponse({"error": "ASTAR_TOKEN not set"}, status_code=400)
    solver = AstarSolver(token)
    return solver.get_rounds()


@app.get("/budget")
async def budget():
    """Utility: check query budget for active round."""
    token = os.environ.get("ASTAR_TOKEN", "")
    if not token:
        return JSONResponse({"error": "ASTAR_TOKEN not set"}, status_code=400)
    solver = AstarSolver(token)
    return solver.get_budget()
