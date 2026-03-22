from pydantic import BaseModel


class FileInput(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileInput] = []
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):
    status: str = "completed"
