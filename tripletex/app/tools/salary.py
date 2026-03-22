from .http_client import TripletexClient

TOOLS = [
    {
        "name": "get_salary_types",
        "description": (
            "Get available salary types (e.g. monthly salary, hourly pay, bonus). "
            "MUST call before create_salary_transaction to find valid salaryType IDs."
        ),
        "input_schema": {"type": "object", "properties": {"count": {"type": "integer", "default": 50}}},
    },
    {
        "name": "create_salary_transaction",
        "description": (
            "Create a salary/payroll transaction for an employee. "
            "Use for 'lønnsutbetaling', 'payroll', 'Gehaltsabrechnung', 'pago de nómina'. "
            "Required: date, year, month, payslips. "
            "Each payslip has employee and specifications (salaryType + amount/rate/count). "
            "Call get_salary_types first to find valid salaryType IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD voucher date"},
                "year": {"type": "integer"},
                "month": {"type": "integer", "description": "1-12"},
                "payslips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee": {"type": "object", "properties": {"id": {"type": "integer"}}},
                            "date": {"type": "string"},
                            "specifications": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "salaryType": {"type": "object", "properties": {"id": {"type": "integer"}}},
                                        "amount": {"type": "number", "description": "Gross salary amount"},
                                        "rate": {"type": "number", "description": "Hourly rate if applicable"},
                                        "count": {"type": "number", "description": "Hours if hourly pay"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "required": ["date", "year", "month", "payslips"],
        },
    },
    {
        "name": "search_salary_transactions",
        "description": "Search for existing salary transactions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "month": {"type": "integer"},
                "count": {"type": "integer", "default": 10},
            },
        },
    },
]


async def execute(name: str, args: dict, client: TripletexClient) -> dict:
    if name == "get_salary_types":
        return await client.get("/salary/type", params={
            "fields": "id,number,name,description",
            "count": args.get("count", 50),
        })

    elif name == "create_salary_transaction":
        return await client.post("/salary/transaction", args)

    elif name == "search_salary_transactions":
        params = {"fields": "id,date,year,month,payslips", "count": args.get("count", 10)}
        if args.get("year"):
            params["year"] = args["year"]
        if args.get("month"):
            params["month"] = args["month"]
        return await client.get("/salary/payslip", params=params)

    return {"error": f"Unknown salary tool: {name}"}
