from .http_client import TripletexClient

TOOLS = [
    {
        "name": "get_accounts",
        "description": (
            "Get chart of accounts. Use to find account numbers/IDs for voucher postings. "
            "Common accounts: 1500=accounts receivable, 2400=accounts payable, "
            "3000=sales revenue, 4000=cost of goods, 1920=bank account."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "number_from": {"type": "integer", "description": "Account number range start"},
                "number_to": {"type": "integer", "description": "Account number range end"},
                "count": {"type": "integer", "default": 100},
            },
        },
    },
    {
        "name": "search_vouchers",
        "description": "Search for vouchers/journal entries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "create_voucher",
        "description": (
            "Create a manual journal entry (voucher) with postings. "
            "Required: date, postings (array of debit/credit lines). "
            "Each posting needs: account ({id or number}), amount, description. "
            "Debits are positive, credits are negative. Total must balance to 0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "description": {"type": "string"},
                "postings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "account": {"type": "object", "properties": {"id": {"type": "integer"}}},
                            "amountCurrency": {"type": "number", "description": "Positive=debit, negative=credit"},
                            "description": {"type": "string"},
                            "customer": {"type": "object", "properties": {"id": {"type": "integer"}}},
                            "supplier": {"type": "object", "properties": {"id": {"type": "integer"}}},
                        },
                        "required": ["account", "amountCurrency"],
                    },
                },
            },
            "required": ["date", "postings"],
        },
    },
    {
        "name": "delete_voucher",
        "description": "Delete a voucher by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "voucher_id": {"type": "integer"},
            },
            "required": ["voucher_id"],
        },
    },
    {
        "name": "get_postings",
        "description": "Get ledger postings for a date range or account.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "account_id": {"type": "integer"},
                "count": {"type": "integer", "default": 20},
            },
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "get_accounts":
        params = {"fields": "id,number,name,type", "count": args.get("count", 100)}
        if args.get("number_from"):
            params["numberFrom"] = args["number_from"]
        if args.get("number_to"):
            params["numberTo"] = args["number_to"]
        return await client.get("/ledger/account", params=params)

    elif name == "search_vouchers":
        params = {"fields": "id,date,description,postings,number", "count": args.get("count", 10)}
        if args.get("date_from"):
            params["dateFrom"] = args["date_from"]
        if args.get("date_to"):
            params["dateTo"] = args["date_to"]
        return await client.get("/ledger/voucher", params=params)

    elif name == "create_voucher":
        return await client.post("/ledger/voucher", args)

    elif name == "delete_voucher":
        return await client.delete(f"/ledger/voucher/{args['voucher_id']}")

    elif name == "get_postings":
        params = {"fields": "id,date,description,account,amountCurrency,voucher", "count": args.get("count", 20)}
        if args.get("date_from"):
            params["dateFrom"] = args["date_from"]
        if args.get("date_to"):
            params["dateTo"] = args["date_to"]
        if args.get("account_id"):
            params["accountId"] = args["account_id"]
        return await client.get("/ledger/posting", params=params)

    return {"error": f"Unknown ledger tool: {name}"}
