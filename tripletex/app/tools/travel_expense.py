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
            "Create a travel expense report. Required: employee, travelDetails object. "
            "Common fields: title, departureDate, returnDate, purpose."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                "title": {"type": "string", "description": "Title/name of the travel expense report"},
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
        "description": (
            "Add a cost/expense line to a travel expense report. "
            "IMPORTANT: paymentType is REQUIRED — call get_travel_payment_types first (travel-specific, NOT invoice payment types). "
            "Use 'costCategory' (object with id) for category. Use 'date' for the date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "travel_expense_id": {"type": "integer"},
                "costCategory": {"type": "object", "properties": {"id": {"type": "integer"}},
                                 "description": "Cost category from get_travel_expense_rates"},
                "amountCurrencyIncVat": {"type": "number"},
                "currency": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "vatType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "paymentType": {"type": "object", "properties": {"id": {"type": "integer"}},
                                "description": "REQUIRED — get from get_travel_payment_types"},
                "comments": {"type": "string", "description": "Optional comment (NOT 'comment')"},
                "date": {"type": "string", "description": "YYYY-MM-DD — date of the cost"},
            },
            "required": ["travel_expense_id", "paymentType"],
        },
    },
    {
        "name": "get_travel_payment_types",
        "description": (
            "Get travel expense payment types (different from invoice payment types). "
            "MUST call this before create_travel_expense_cost to get valid paymentType IDs."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_travel_expenses":
        params = {"fields": "id,title,displayName,employee,travelDetails,isApproved", "count": args.get("count", 10)}
        if args.get("employee_id"):
            params["employeeId"] = args["employee_id"]
        return await client.get("/travelExpense", params=params)

    elif name == "create_travel_expense":
        return await client.post("/travelExpense", args)

    elif name == "delete_travel_expense":
        return await client.delete(f"/travelExpense/{args['travel_expense_id']}")

    elif name == "get_travel_expense_rates":
        return await client.get("/travelExpense/costCategory", params={"fields": "id,description,displayName", "count": args.get("count", 50)})

    elif name == "get_travel_payment_types":
        return await client.get("/travelExpense/paymentType", params={"fields": "id,description,displayName", "count": 50})

    elif name == "create_travel_expense_cost":
        travel_expense_id = args.pop("travel_expense_id")
        # Remap any wrong field names the model might still use
        if "category" in args and isinstance(args["category"], dict):
            args["costCategory"] = args.pop("category")
        if "rateDate" in args:
            args["date"] = args.pop("rateDate")
        if "costDate" in args:
            args["date"] = args.pop("costDate")
        if "comment" in args:
            args["comments"] = args.pop("comment")
        body = {"travelExpense": {"id": travel_expense_id}, **args}
        return await client.post("/travelExpense/cost", body)

    return {"error": f"Unknown travel expense tool: {name}"}
