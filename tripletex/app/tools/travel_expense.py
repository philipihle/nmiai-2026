from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_travel_expenses",
        "description": "Search for travel expense reports.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "create_travel_expense",
        "description": (
            "Create a travel expense report. Required: employee, traveller, travelDetails object. "
            "Common fields: description, departureDate, returnDate, purpose."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                "description": {"type": "string"},
                "travelDetails": {
                    "type": "object",
                    "properties": {
                        "isForeignTravel": {"type": "boolean"},
                        "departureDate": {"type": "string", "description": "YYYY-MM-DD"},
                        "returnDate": {"type": "string"},
                        "detailedJourneyDescription": {"type": "string"},
                        "departureFrom": {"type": "string"},
                        "destination": {"type": "string"},
                        "departureTime": {"type": "string"},
                        "returnTime": {"type": "string"},
                        "purpose": {"type": "string"},
                    },
                },
                "voucher": {"type": "object", "properties": {"date": {"type": "string"}}},
            },
            "required": ["employee"],
        },
    },
    {
        "name": "delete_travel_expense",
        "description": "Delete a travel expense report by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "travel_expense_id": {"type": "integer"},
            },
            "required": ["travel_expense_id"],
        },
    },
    {
        "name": "get_travel_expense_rates",
        "description": "Get available travel expense rate categories (for per-diem, mileage, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "create_travel_expense_cost",
        "description": "Add a cost/expense line to a travel expense report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "travel_expense_id": {"type": "integer"},
                "costType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "amountCurrencyIncVat": {"type": "number"},
                "currency": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "vatType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "comment": {"type": "string"},
                "costDate": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["travel_expense_id"],
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_travel_expenses":
        params = {"fields": "id,description,employee,travelDetails,isApproved", "count": args.get("count", 10)}
        if args.get("employee_id"):
            params["employeeId"] = args["employee_id"]
        return await client.get("/travelExpense", params=params)

    elif name == "create_travel_expense":
        return await client.post("/travelExpense", args)

    elif name == "delete_travel_expense":
        return await client.delete(f"/travelExpense/{args['travel_expense_id']}")

    elif name == "get_travel_expense_rates":
        return await client.get("/travelExpense/costType", params={"fields": "id,name,description", "count": args.get("count", 50)})

    elif name == "create_travel_expense_cost":
        travel_expense_id = args.pop("travel_expense_id")
        body = {"travelExpense": {"id": travel_expense_id}, **args}
        return await client.post("/travelExpense/cost", body)

    return {"error": f"Unknown travel expense tool: {name}"}
