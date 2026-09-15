from fastapi import FastAPI

from app.schemas.chat import ChatRequest

app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "llm-chat-api", "version": app.version}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Temporary echo endpoint: proves request validation end to end.
# Moves to app/api/chat.py once the service layer exists.
@app.post("/v1/chat")
async def chat(request: ChatRequest) -> ChatRequest:
    return request
