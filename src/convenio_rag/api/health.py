"""Health check endpoint, used by Docker and uptime monitors."""

from fastapi import APIRouter
from pydantic import BaseModel

from convenio_rag import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
