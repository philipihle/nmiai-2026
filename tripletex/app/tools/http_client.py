import httpx


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self._auth = ("0", session_token)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(auth=self._auth, timeout=30.0)

    def _parse(self, r: httpx.Response) -> dict:
        try:
            return r.json() if r.content else {}
        except Exception:
            return {"_raw": r.text[:200]} if r.text else {}

    async def get(self, path: str, params: dict | None = None) -> dict:
        async with self._client() as c:
            r = await c.get(f"{self.base_url}{path}", params=params)
            return {"status_code": r.status_code, "body": self._parse(r)}

    async def post(self, path: str, body: dict) -> dict:
        async with self._client() as c:
            r = await c.post(f"{self.base_url}{path}", json=body)
            return {"status_code": r.status_code, "body": self._parse(r)}

    async def put(self, path: str, body: dict) -> dict:
        async with self._client() as c:
            r = await c.put(f"{self.base_url}{path}", json=body)
            return {"status_code": r.status_code, "body": self._parse(r)}

    async def delete(self, path: str) -> dict:
        async with self._client() as c:
            r = await c.delete(f"{self.base_url}{path}")
            return {"status_code": r.status_code, "body": self._parse(r)}
