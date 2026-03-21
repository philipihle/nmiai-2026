from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_projects",
        "description": "Search for projects by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "create_project",
        "description": (
            "Create a new project. Required: name, startDate. "
            "Common optional fields: customer ({id}), projectManager ({id}), endDate, "
            "description, number (project number), mainProject ({id})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "startDate": {"type": "string", "description": "YYYY-MM-DD"},
                "endDate": {"type": "string"},
                "number": {"type": "string"},
                "description": {"type": "string"},
                "customer": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "projectManager": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "mainProject": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["name", "startDate"],
        },
    },
    {
        "name": "search_departments",
        "description": "Search for departments by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "create_department",
        "description": "Create a new department. Required: name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "departmentNumber": {"type": "string"},
                "departmentManager": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["name"],
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_projects":
        params = {"fields": "id,name,number,customer,startDate,endDate,projectManager", "count": args.get("count", 10)}
        if args.get("name"):
            params["name"] = args["name"]
        return await client.get("/project", params=params)

    elif name == "create_project":
        return await client.post("/project", args)

    elif name == "search_departments":
        params = {"fields": "id,name,departmentNumber,departmentManager", "count": args.get("count", 20)}
        if args.get("name"):
            params["name"] = args["name"]
        return await client.get("/department", params=params)

    elif name == "create_department":
        return await client.post("/department", args)

    return {"error": f"Unknown project tool: {name}"}
