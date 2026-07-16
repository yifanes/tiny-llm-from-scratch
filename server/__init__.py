"""OpenAI-compatible HTTP service for TinyGPT."""

from .app import app, create_app

__all__ = ["app", "create_app"]
