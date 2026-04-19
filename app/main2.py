"""Compatibility entrypoint.

Use app.main:app for new deployments. This module remains so older commands
such as `uvicorn app.main2:app` still start the same application.
"""
from app.main import app
