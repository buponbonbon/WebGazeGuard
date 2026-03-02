from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .db import init_db
from .routers import auth, me, analysis, chat, health

app = FastAPI(
    title="WebGazeGuard API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(me.router, prefix="/api/me", tags=["me"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
