"""
Middleware - Rate limiting
"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_history = {}

    EXEMPT_PREFIXES = (
        "/api/v1/health",
        "/api/v1/notifications",
        "/api/v1/tutor",
        "/api/v1/dashboard",
        "/api/v1/analytics",
    )

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        # Always pass OPTIONS (preflight) straight through so CORS middleware
        # can handle it — blocking OPTIONS would cause CORS errors.
        if request.method == "OPTIONS":
            return await call_next(request)

        if any(request.url.path.startswith(p) for p in self.EXEMPT_PREFIXES):
            return await call_next(request)

        forwarded = request.headers.get("x-forwarded-for")
        client_host = (
            forwarded.split(",")[0].strip()
            if forwarded
            else (request.client.host if request.client else "unknown")
        )
        current_time = time.time()

        if client_host not in self.request_history:
            self.request_history[client_host] = []

        self.request_history[client_host] = [
            t for t in self.request_history[client_host]
            if current_time - t < 60
        ]

        if len(self.request_history[client_host]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_host}")
            # Use JSONResponse (NOT raise HTTPException) so this response
            # still passes through CORSMiddleware and gets the
            # Access-Control-Allow-Origin header attached.
            origin = request.headers.get("origin", "")
            allowed = settings.ALLOWED_ORIGINS
            cors_origin = origin if (origin in allowed or "*" in allowed) else ""
            headers = {}
            if cors_origin:
                headers["Access-Control-Allow-Origin"] = cors_origin
                headers["Access-Control-Allow-Credentials"] = "true"
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers=headers,
            )

        self.request_history[client_host].append(current_time)
        return await call_next(request)
