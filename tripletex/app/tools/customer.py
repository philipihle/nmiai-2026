from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_customers",
        "description": "Search for customers by name. Always call before creating to avoid duplicates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Customer name or partial name"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "create_customer",
        "description": (
            "Create a new customer. Required: name. "
            "Common fields: email, phoneNumber, address (object with addressLine1, postalCode, city, country {id}), "
            "organizationNumber, isCustomer (bool, default true), isSupplier (bool)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumber": {"type": "string"},
                "organizationNumber": {"type": "string"},
                "isCustomer": {"type": "boolean", "default": True},
                "isSupplier": {"type": "boolean"},
                "invoiceEmail": {"type": "string", "description": "Email for invoices (separate from main email)"},
                "language": {"type": "string", "description": "NO or EN"},
                "physicalAddress": {
                    "type": "object",
                    "description": "Physical/visiting address",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                        "country": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    },
                },
                "postalAddress": {
                    "type": "object",
                    "description": "Postal address (use if task specifies postal address)",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                        "country": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    },
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "update_customer",
        "description": "Update an existing customer by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumber": {"type": "string"},
                "organizationNumber": {"type": "string"},
                "physicalAddress": {"type": "object"},
                "postalAddress": {"type": "object"},
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_customer",
        "description": "Get a single customer by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
            },
            "required": ["customer_id"],
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_customers":
        params = {
            "fields": "id,name,email,phoneNumber,organizationNumber,physicalAddress,postalAddress",
            "count": args.get("count", 10),
        }
        if args.get("name"):
            params["customerName"] = args["name"]
        return await client.get("/customer", params=params)

    elif name == "create_customer":
        if "isCustomer" not in args:
            args["isCustomer"] = True
        return await client.post("/customer", args)

    elif name == "update_customer":
        customer_id = args.pop("customer_id")
        # Fetch current to get version and merge fields
        current = await client.get(f"/customer/{customer_id}", params={"fields": "*"})
        if current["status_code"] == 200:
            existing = current["body"].get("value", {})
            update_body = {**existing, **args, "id": customer_id}
            return await client.put(f"/customer/{customer_id}", update_body)
        args["id"] = customer_id
        return await client.put(f"/customer/{customer_id}", args)

    elif name == "get_customer":
        return await client.get(f"/customer/{args['customer_id']}", params={"fields": "*"})

    return {"error": f"Unknown customer tool: {name}"}
