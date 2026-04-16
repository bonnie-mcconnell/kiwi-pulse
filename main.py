"""
main.py

Entry point for the KiwiPulse API.
"""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="KiwiPulse",
    description="Probabilistic Market Intelligence Engine - "
                "Bayesian inference over noisy sentiment signals.",
    version="0.1.0",
)

app.include_router(router)