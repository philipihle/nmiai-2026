from .http_client import TripletexClient

TOOLS = [
    {
        "name": "create_timesheet_entry",
        "description": (
            "Create a time registration / timesheet entry for an employee. "
            "Required: employee, activity, date, hours. "
            "You MUST first search for activities using get_timesheet_activities to find the activity ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                "activity": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"],
                             "description": "Activity reference — get ID from get_timesheet_activities"},
                "project": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "hours": {"type": "number", "description": "Number of hours worked"},
                "comment": {"type": "string"},
            },
            "required": ["employee", "activity", "date", "hours"],
        },
    },
    {
        "name": "search_timesheet_entries",
        "description": "Search for existing timesheet entries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "project_id": {"type": "integer"},
                "count": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "get_timesheet_activities",
        "description": (
            "Get available timesheet activities. MUST call this before creating timesheet entries "
            "to find valid activity IDs. Can filter by project ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer", "description": "Filter by project"},
                "count": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "update_timesheet_entry",
        "description": "Update an existing timesheet entry by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "integer"},
                "hours": {"type": "number"},
                "date": {"type": "string"},
                "comment": {"type": "string"},
                "activity": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["entry_id"],
        },
    },
    {
        "name": "delete_timesheet_entry",
        "description": "Delete a timesheet entry by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "integer"},
            },
            "required": ["entry_id"],
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "create_timesheet_entry":
        return await client.post("/timesheet/entry", args)

    elif name == "search_timesheet_entries":
        params = {
            "fields": "id,date,hours,employee,activity,project,comment",
            "count": args.get("count", 20),
        }
        if args.get("employee_id"):
            params["employeeId"] = args["employee_id"]
        if args.get("date_from"):
            params["dateFrom"] = args["date_from"]
        if args.get("date_to"):
            params["dateTo"] = args["date_to"]
        if args.get("project_id"):
            params["projectId"] = args["project_id"]
        return await client.get("/timesheet/entry", params=params)

    elif name == "get_timesheet_activities":
        params = {
            "fields": "id,name,number,displayName",
            "count": args.get("count", 50),
        }
        if args.get("project_id"):
            params["projectId"] = args["project_id"]
        return await client.get("/activity", params=params)

    elif name == "update_timesheet_entry":
        entry_id = args.pop("entry_id")
        # Need to get current version first
        current = await client.get(f"/timesheet/entry/{entry_id}", params={"fields": "*"})
        if current["status_code"] == 200:
            entry = current["body"].get("value", {})
            update_body = {**entry, **args, "id": entry_id}
            return await client.put(f"/timesheet/entry/{entry_id}", update_body)
        return current

    elif name == "delete_timesheet_entry":
        return await client.delete(f"/timesheet/entry/{args['entry_id']}")

    return {"error": f"Unknown timesheet tool: {name}"}
