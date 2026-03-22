from .http_client import TripletexClient
from . import employee, customer, product, invoice, travel_expense, project, ledger, timesheet, salary

_MODULES = [employee, customer, product, invoice, travel_expense, project, ledger, timesheet, salary]

_TOOL_MAP: dict[str, object] = {}
for _mod in _MODULES:
    for _tool in _mod.TOOLS:
        _TOOL_MAP[_tool["name"]] = _mod


def get_all_tools() -> list[dict]:
    tools = []
    for mod in _MODULES:
        tools.extend(mod.TOOLS)
    return tools


async def dispatch(name: str, args: dict, client: TripletexClient) -> dict:
    mod = _TOOL_MAP.get(name)
    if mod is None:
        return {"error": f"Unknown tool: {name}"}
    return await mod.execute(name, args, client)
