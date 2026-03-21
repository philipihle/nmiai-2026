from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_products",
        "description": "Search for products by name or number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "number": {"type": "string", "description": "Product number/SKU"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "create_product",
        "description": (
            "Create a new product. Required: name. "
            "Common fields: number (SKU), costExcludingVatCurrency (price), "
            "priceExcludingVatCurrency, priceIncludingVatCurrency, vatType ({id}), unit ({id})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "number": {"type": "string"},
                "description": {"type": "string"},
                "costExcludingVatCurrency": {"type": "number"},
                "priceExcludingVatCurrency": {"type": "number"},
                "priceIncludingVatCurrency": {"type": "number"},
                "vatType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "unit": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["name"],
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_products":
        params = {"fields": "id,name,number,priceExcludingVatCurrency,vatType", "count": args.get("count", 10)}
        if args.get("name"):
            params["name"] = args["name"]
        if args.get("number"):
            params["number"] = args["number"]
        return await client.get("/product", params=params)

    elif name == "create_product":
        return await client.post("/product", args)

    return {"error": f"Unknown product tool: {name}"}
