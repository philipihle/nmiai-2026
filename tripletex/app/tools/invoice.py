from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_orders",
        "description": "Search for existing orders. orderDateFrom and orderDateTo are REQUIRED by the API (use wide range like '2020-01-01' to '2030-12-31' if unsure).",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "orderDateFrom": {"type": "string", "description": "YYYY-MM-DD — REQUIRED"},
                "orderDateTo": {"type": "string", "description": "YYYY-MM-DD — REQUIRED"},
                "count": {"type": "integer", "default": 10},
            },
            "required": ["orderDateFrom", "orderDateTo"],
        },
    },
    {
        "name": "create_order",
        "description": (
            "Create a new order for a customer. Required: orderDate, deliveryDate, customer. "
            "Optionally include orderLines array. An order is required before creating an invoice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "orderDate": {"type": "string", "description": "YYYY-MM-DD"},
                "deliveryDate": {"type": "string", "description": "YYYY-MM-DD — REQUIRED, use same as orderDate if not specified"},
                "customer": {"type": "object", "properties": {"id": {"type": "integer"}}, "description": "Customer reference {id: N}"},
                "orderLines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "object", "properties": {"id": {"type": "integer"}}},
                            "description": {"type": "string"},
                            "count": {"type": "number"},
                            "unitPriceExcludingVatCurrency": {"type": "number"},
                            "vatType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                        },
                    },
                },
            },
            "required": ["orderDate", "deliveryDate", "customer"],
        },
    },
    {
        "name": "add_order_line",
        "description": "Add a product line to an existing order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "product": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "description": {"type": "string"},
                "count": {"type": "number", "description": "Quantity"},
                "unitPriceExcludingVatCurrency": {"type": "number"},
                "vatType": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "create_invoice",
        "description": (
            "Create an invoice from an existing order. Required: invoiceDate, customer, orders (list of order IDs). "
            "invoiceDueDate defaults to 14 days after invoiceDate if not specified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "invoiceDate": {"type": "string", "description": "YYYY-MM-DD"},
                "invoiceDueDate": {"type": "string", "description": "YYYY-MM-DD"},
                "customer": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                "orders": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    "description": "List of order references [{id: N}]",
                },
                "comment": {"type": "string"},
            },
            "required": ["invoiceDate", "customer", "orders"],
        },
    },
    {
        "name": "search_invoices",
        "description": "Search for invoices. invoiceDateFrom and invoiceDateTo are REQUIRED by the API (use wide range like '2020-01-01' to '2030-12-31' if unsure).",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "invoice_date_from": {"type": "string", "description": "YYYY-MM-DD — REQUIRED"},
                "invoice_date_to": {"type": "string", "description": "YYYY-MM-DD — REQUIRED"},
                "count": {"type": "integer", "default": 10},
            },
            "required": ["invoice_date_from", "invoice_date_to"],
        },
    },
    {
        "name": "create_invoice_payment",
        "description": "Register a payment on an invoice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "integer"},
                "paymentDate": {"type": "string", "description": "YYYY-MM-DD"},
                "paymentTypeId": {"type": "integer", "description": "Payment type ID (usually 1 for bank)"},
                "paidAmount": {"type": "number", "description": "Amount paid"},
            },
            "required": ["invoice_id", "paymentDate", "paymentTypeId", "paidAmount"],
        },
    },
    {
        "name": "create_credit_note",
        "description": "Create a credit note for an invoice. This reverses/credits the invoice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "integer"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "comment": {"type": "string"},
            },
            "required": ["invoice_id", "date"],
        },
    },
    {
        "name": "get_payment_types",
        "description": "List available payment types. Call once to find the right paymentTypeId.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "search_orders":
        params = {"fields": "id,orderDate,deliveryDate,customer,orderLines", "count": args.get("count", 10)}
        if args.get("customer_id"):
            params["customerId"] = args["customer_id"]
        params["orderDateFrom"] = args.get("orderDateFrom", "2020-01-01")
        params["orderDateTo"] = args.get("orderDateTo", "2030-12-31")
        return await client.get("/order", params=params)

    elif name == "create_order":
        return await client.post("/order", args)

    elif name == "add_order_line":
        order_id = args.pop("order_id")
        body = {"order": {"id": order_id}, **args}
        return await client.post("/order/orderline", body)

    elif name == "create_invoice":
        # invoiceDueDate is required by API — default to 14 days after invoiceDate
        if "invoiceDueDate" not in args and "invoiceDate" in args:
            from datetime import date, timedelta
            invoice_date = date.fromisoformat(args["invoiceDate"])
            args["invoiceDueDate"] = (invoice_date + timedelta(days=14)).isoformat()
        return await client.post("/invoice", args)

    elif name == "search_invoices":
        params = {"fields": "id,invoiceDate,invoiceDueDate,customer,amountCurrency,amountOutstandingTotal,isCredited", "count": args.get("count", 10)}
        if args.get("customer_id"):
            params["customerId"] = args["customer_id"]
        params["invoiceDateFrom"] = args.get("invoice_date_from", "2020-01-01")
        params["invoiceDateTo"] = args.get("invoice_date_to", "2030-12-31")
        return await client.get("/invoice", params=params)

    elif name == "create_invoice_payment":
        invoice_id = args.pop("invoice_id")
        import httpx
        async with httpx.AsyncClient(auth=client._auth, timeout=30.0) as c:
            r = await c.put(
                f"{client.base_url}/invoice/{invoice_id}/:payment",
                params={
                    "paymentDate": args["paymentDate"],
                    "paymentTypeId": args["paymentTypeId"],
                    "paidAmount": args["paidAmount"],
                },
            )
            return {"status_code": r.status_code, "body": r.json() if r.content else {}}

    elif name == "create_credit_note":
        invoice_id = args.pop("invoice_id")
        params = {"date": args["date"]}
        if args.get("comment"):
            params["comment"] = args["comment"]
        async with __import__("httpx").AsyncClient(auth=client._auth, timeout=30.0) as c:
            r = await c.put(
                f"{client.base_url}/invoice/{invoice_id}/:createCreditNote",
                params=params,
            )
            return {"status_code": r.status_code, "body": r.json() if r.content else {}}

    elif name == "get_payment_types":
        return await client.get("/invoice/paymentType", params={"fields": "id,description,displayName", "count": 100})

    return {"error": f"Unknown invoice tool: {name}"}
