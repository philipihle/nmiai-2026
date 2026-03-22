import logging
from .http_client import TripletexClient

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "name": "setup_invoice_bank_account",
        "description": (
            "MUST be called before creating any invoice. "
            "Sets the company bank account number on ledger account 1920, which is required for invoice creation. "
            "Use a realistic Norwegian bank account number (11 digits)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bank_account_number": {
                    "type": "string",
                    "description": "11-digit Norwegian bank account number. Use '15001234568' as default. Must pass Norwegian MOD11 validation.",
                },
            },
            "required": ["bank_account_number"],
        },
    },
    {
        "name": "get_accounts",
        "description": (
            "Get chart of accounts. Use to find account IDs for voucher postings. "
            "Filter by specific account numbers using the 'numbers' parameter (comma-separated). "
            "Common accounts: 1500=receivable, 2400=payable, 3000=sales, 4000=COGS, 1920=bank."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "numbers": {"type": "string", "description": "Comma-separated account numbers to look up, e.g. '1920,3000'"},
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
        "name": "get_vat_types",
        "description": "Get available VAT types. Use to find vatType IDs for products, orders, and voucher postings. typeOfVat: OUTGOING (for sales/products), INCOMING (for purchases).",
        "input_schema": {
            "type": "object",
            "properties": {
                "typeOfVat": {"type": "string", "description": "OUTGOING or INCOMING"},
            },
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
    if name == "setup_invoice_bank_account":
        # Find account 1920
        accounts = await client.get("/ledger/account", params={"number": "1920", "fields": "id,version,number,name"})
        values = accounts.get("body", {}).get("values", [])
        if not values:
            return {"error": "Could not find account 1920"}
        account = values[0]
        account_id = account["id"]
        # Update with bank account number
        return await client.put(f"/ledger/account/{account_id}", {
            "id": account_id,
            "version": account.get("version", 0),
            "number": 1920,
            "name": account.get("name", "Bankinnskudd"),
            "bankAccountNumber": args["bank_account_number"],
            "isBankAccount": True,
            "isInvoiceAccount": True,
        })
    if name == "get_accounts":
        params = {"fields": "id,number,name,type", "count": args.get("count", 100)}
        if args.get("numbers"):
            params["number"] = args["numbers"]
        return await client.get("/ledger/account", params=params)

    elif name == "search_vouchers":
        params = {"fields": "id,date,description,number,postings(id,account,amountCurrency,description)", "count": args.get("count", 100)}
        if args.get("date_from"):
            params["dateFrom"] = args["date_from"]
        if args.get("date_to"):
            params["dateTo"] = args["date_to"]
        return await client.get("/ledger/voucher", params=params)

    elif name == "create_voucher":
        if "postings" in args:
            voucher_date = args.get("date")
            clean_postings = []
            for posting in args["postings"]:
                # Accept amountCurrency or amount as input
                val = posting.get("amountCurrency")
                if val is None:
                    val = posting.get("amount")
                if val is None:
                    val = posting.get("amountGross", 0)
                # Build a minimal clean posting — only fields Tripletex accepts
                clean = {
                    "account": posting["account"],
                    "amountCurrency": val,
                    "date": posting.get("date") or voucher_date,
                }
                if posting.get("description"):
                    clean["description"] = posting["description"]
                if posting.get("customer"):
                    clean["customer"] = posting["customer"]
                if posting.get("supplier"):
                    clean["supplier"] = posting["supplier"]
                if posting.get("vatType"):
                    clean["vatType"] = posting["vatType"]
                clean_postings.append(clean)
            args["postings"] = clean_postings
        import json as _json
        logger.info(f"create_voucher sending: {_json.dumps(args, ensure_ascii=False, default=str)[:500]}")
        # sendToLedger=true ensures voucher is committed to the ledger
        import httpx as _httpx
        async with _httpx.AsyncClient(auth=client._auth, timeout=30.0) as c:
            r = await c.post(
                f"{client.base_url}/ledger/voucher",
                json=args,
                params={"sendToLedger": "true"},
            )
            body = r.json() if r.content else {}
            # If systemgenererte error, look up account numbers for clearer feedback
            if r.status_code == 422:
                msgs = body.get("validationMessages", [])
                if any("systemgenererte" in (m.get("message") or "") for m in msgs):
                    account_info = []
                    for posting in clean_postings:
                        acc_id = posting["account"]["id"]
                        try:
                            ar = await c.get(f"{client.base_url}/ledger/account/{acc_id}", params={"fields": "id,number,name"})
                            if ar.status_code == 200:
                                acc = ar.json().get("value", {})
                                account_info.append(f"{acc.get('number')} ({acc.get('name')})")
                        except Exception:
                            account_info.append(str(acc_id))
                    body["_system_managed_accounts"] = account_info
                    body["_hint"] = f"These accounts are system-managed in this environment and CANNOT be used in manual vouchers: {', '.join(account_info)}. Try different accounts."
                    logger.warning(f"create_voucher systemgenererte for accounts: {account_info}")
            result = {"status_code": r.status_code, "body": body}
            logger.info(f"create_voucher response: {_json.dumps(result, ensure_ascii=False, default=str)[:500]}")
            return result

    elif name == "delete_voucher":
        return await client.delete(f"/ledger/voucher/{args['voucher_id']}")

    elif name == "get_vat_types":
        params = {"fields": "id,name,number,percentage,displayName"}
        if args.get("typeOfVat"):
            params["typeOfVat"] = args["typeOfVat"]
        return await client.get("/ledger/vatType", params=params)

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
