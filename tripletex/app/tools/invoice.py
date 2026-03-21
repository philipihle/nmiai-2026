from .http_client import TripletexClient

TOOLS = [
    {
        "name": "search_orders",
        "description": "Search for existing orders. Use to find orders before creating invoices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "create_order",
        "description": (
            "Create a new order for a customer. Required: orderDate, customer. "
            "Optionally include orderLines array. An order is required before creating an invoice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "orderDate": {"type": "string", "description": "YYYY-MM-DD"},
                "customer": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                "deliveryDate": {"type": "string"},
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
            "required": ["orderDate", "customer"],
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
        "description": "Search for invoices. Useful for finding invoices to register payment on or create credit notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "invoice_date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "invoice_date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "count": {"type": "integer", "default": 10},
            },
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
        params = {"fields": "id,orderDate,customer,orderLines,totalAmountExcludingVat", "count": args.get("count", 10)}
        if args.get("customer_id"):
            params["customerId"] = args["customer_id"]
        return await client.get("/order", params=params)

    elif name == "create_order":
        return await client.post("/order", args)

    elif name == "add_order_line":
        order_id = args.pop("order_id")
        body = {"order": {"id": order_id}, **args}
        return await client.post("/order/orderline", body)

    elif name == "create_invoice":
        return await client.post("/invoice", args)

    elif name == "search_invoices":
        params = {"fields": "id,invoiceDate,invoiceDueDate,customer,amountCurrency,amountOutstandingCurrency", "count": args.get("count", 10)}
        if args.get("customer_id"):
            params["customerId"] = args["customer_id"]
        if args.get("invoice_date_from"):
            params["invoiceDateFrom"] = args["invoice_date_from"]
        if args.get("invoice_date_to"):
            params["invoiceDateTo"] = args["invoice_date_to"]
        return await client.get("/invoice", params=params)

    elif name == "create_invoice_payment":
        invoice_id = args.pop("invoice_id")
        import httpx
        async with httpx.AsyncClient(auth=client._auth, timeout=30.0) as c:
            r = await c.post(
                f"{client.base_url}/invoice/{invoice_id}/:pay",
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
        # Credit note: POST /invoice/{id}/:createCreditNote
        async with __import__("httpx").AsyncClient(auth=client._auth, timeout=30.0) as c:
            r = await c.post(
                f"{client.base_url}/invoice/{invoice_id}/:createCreditNote",
                params=params,
            )
            return {"status_code": r.status_code, "body": r.json() if r.content else {}}

    elif name == "get_payment_types":
        return await client.get("/invoice/paymentType", params={"fields": "id,name", "count": 100})

    return {"error": f"Unknown invoice tool: {name}"}
