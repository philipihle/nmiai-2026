from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_employees",
        "description": "Search for employees by name or employee number. Always call this before creating to avoid duplicates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name or partial name to search"},
                "count": {"type": "integer", "description": "Max results (default 10)", "default": 10},
            },
        },
    },
    {
        "name": "get_employee",
        "description": "Get a single employee by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer"},
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "create_employee",
        "description": (
            "Create a new employee. Required: firstName, lastName. "
            "Common optional fields: email, phoneNumberMobile, startDate (YYYY-MM-DD), "
            "employeeNumber, department ({id}), position ({id}). "
            "To set administrator role use the set_employee_role tool after creation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumberMobile": {"type": "string"},
                "startDate": {"type": "string", "description": "YYYY-MM-DD"},
                "employeeNumber": {"type": "string"},
                "department": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
                "position": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            },
            "required": ["firstName", "lastName"],
        },
    },
    {
        "name": "update_employee",
        "description": "Update an existing employee by ID. Only include fields you want to change.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer"},
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumberMobile": {"type": "string"},
                "startDate": {"type": "string"},
                "department": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "get_employee_employment",
        "description": "Get employment details for an employee (roles, positions, employment type).",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer"},
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "create_employment",
        "description": "Create an employment record for an employee.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer"},
                "startDate": {"type": "string", "description": "YYYY-MM-DD"},
                "employmentType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "remunerationType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "workingHoursScheme": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["employee_id", "startDate"],
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_employees":
        params = {"fields": "id,firstName,lastName,email,phoneNumberMobile,employeeNumber,department"}
        if args.get("query"):
            params["firstName"] = args["query"]
        params["count"] = args.get("count", 10)
        result = await client.get("/employee", params=params)
        # Also try lastName search if firstName returns nothing
        if result["status_code"] == 200 and not result["body"].get("values"):
            params2 = dict(params)
            params2.pop("firstName", None)
            params2["lastName"] = args["query"]
            result2 = await client.get("/employee", params=params2)
            if result2["status_code"] == 200 and result2["body"].get("values"):
                return result2
        return result

    elif name == "get_employee":
        return await client.get(f"/employee/{args['employee_id']}", params={"fields": "*"})

    elif name == "create_employee":
        return await client.post("/employee", args)

    elif name == "update_employee":
        emp_id = args.pop("employee_id")
        args["id"] = emp_id
        return await client.put(f"/employee/{emp_id}", args)

    elif name == "get_employee_employment":
        return await client.get(
            "/employee/employment",
            params={"employeeId": args["employee_id"], "fields": "*"},
        )

    elif name == "create_employment":
        emp_id = args.pop("employee_id")
        body = {"employee": {"id": emp_id}, **args}
        return await client.post("/employee/employment", body)

    return {"error": f"Unknown employee tool: {name}"}
