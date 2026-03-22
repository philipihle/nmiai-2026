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
            "IMPORTANT: Always include userType to avoid validation errors. "
            "userType values: 'STANDARD' (default), 'EXTENDED' (if task says admin/kontoadministrator/administrator), 'NO_ACCESS'. "
            "Common optional fields: email, phoneNumberMobile, startDate (YYYY-MM-DD), "
            "employeeNumber, department ({id})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumberMobile": {"type": "string"},
                "dateOfBirth": {"type": "string", "description": "YYYY-MM-DD — required before creating employment"},
                "employeeNumber": {"type": "string"},
                "userType": {
                    "type": "string",
                    "description": "STANDARD (default, limited access), EXTENDED (full access, use if task says admin/administrator), NO_ACCESS (no login)",
                },
                "department": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            },
            "required": ["firstName", "lastName", "userType", "department"],
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
        "description": (
            "Create an employment record for an employee. Use this to set startDate. "
            "NOTE: employee must have dateOfBirth set first (use update_employee if not set during creation). "
            "Required: employee_id, startDate."
        ),
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
        fields = "id,firstName,lastName,email,phoneNumberMobile,employeeNumber,department"
        query = args.get("query", "")
        count = args.get("count", 10)

        # Try email search first if query looks like an email
        if "@" in query:
            r = await client.get("/employee", params={"email": query, "fields": fields, "count": count})
            if r["status_code"] == 200 and r["body"].get("values"):
                return r

        # Try firstName
        r = await client.get("/employee", params={"firstName": query, "fields": fields, "count": count})
        if r["status_code"] == 200 and r["body"].get("values"):
            return r

        # Try lastName
        r2 = await client.get("/employee", params={"lastName": query, "fields": fields, "count": count})
        if r2["status_code"] == 200 and r2["body"].get("values"):
            return r2

        return r

    elif name == "get_employee":
        return await client.get(f"/employee/{args['employee_id']}", params={"fields": "*"})

    elif name == "create_employee":
        result = await client.post("/employee", args)
        # If email already exists, find and return the existing employee
        if result["status_code"] == 422:
            msgs = result["body"].get("validationMessages") or []
            email_dup = any("e-postadressen" in (m.get("message") or "") for m in msgs)
            if email_dup and args.get("email"):
                r = await client.get("/employee", params={"email": args["email"], "fields": "id,firstName,lastName,email,department"})
                if r["status_code"] == 200 and r["body"].get("values"):
                    existing = r["body"]["values"][0]
                    return {"status_code": 200, "body": {"value": existing, "_note": "Employee already exists, returned existing"}}
        return result

    elif name == "update_employee":
        emp_id = args.pop("employee_id")
        # Read-only fields Tripletex rejects on PUT
        READ_ONLY = {"changes", "url", "changesWithImplications", "createdDate",
                     "lastModifiedDate", "employeeNumber", "isPrivateAddress"}
        # Fetch current employee to get version and merge fields
        current = await client.get(f"/employee/{emp_id}", params={"fields": "*"})
        if current["status_code"] == 200:
            existing = current["body"].get("value", {})
            # Remove read-only fields from existing before merging
            existing_clean = {k: v for k, v in existing.items() if k not in READ_ONLY and v is not None}
            update_body = {**existing_clean, **args, "id": emp_id}
            return await client.put(f"/employee/{emp_id}", update_body)
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
