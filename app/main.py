from fastapi import FastAPI

from app.api.chat import router as chat_router

app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
)

app.include_router(chat_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "llm-chat-api", "version": app.version}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
