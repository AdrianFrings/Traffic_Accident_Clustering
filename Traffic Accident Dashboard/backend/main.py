"""FastAPI backend for the Traffic Accident Risk Dashboard.

Serves precomputed clustering results as JSON so the frontend stays fully
client-side and snappy -- no per-request geocoding, no full-page reruns, and
the heavy geo/ML stack lives only in the offline processor, not here.

In production it also serves the built frontend (static bundle) so the whole
app runs from a single small container, same-origin (the frontend calls /api).
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import data_access as dao

app = FastAPI(title="Traffic Accident Risk API", version="1.0")

# Allow the (separately served) frontend dev server to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/cities")
def cities() -> dict:
    return {"cities": dao.list_cities()}


@app.get("/api/cities/{city}/modes")
def modes(city: str) -> dict:
    available = dao.list_modes(city)
    if not available:
        raise HTTPException(status_code=404, detail=f"No data for city '{city}'.")
    return {"city": city, "modes": available}


@app.get("/api/clusters")
def clusters(
    city: str = Query(..., description="City name"),
    mode: str = Query(..., description="Transport mode: Car, Bicycle, Pedestrian, All"),
) -> dict:
    result = dao.get_clusters(city, mode)
    return {"city": city, "mode": mode, "count": len(result), "clusters": result}


@app.get("/api/clusters/{cluster_id}/points")
def cluster_points(
    cluster_id: int,
    city: str = Query(...),
    mode: str = Query(...),
) -> dict:
    result = dao.get_cluster_points(city, mode, cluster_id)
    if not result["cluster"]:
        raise HTTPException(
            status_code=404,
            detail=f"No points for cluster {cluster_id} in {city}/{mode}.",
        )
    return {
        "city": city,
        "mode": mode,
        "cluster_id": cluster_id,
        **result,
    }


# Serve the built frontend (if present) at the root. Mounted last so the /api
# routes above take precedence. In local dev the frontend runs on its own Vite
# server, so this directory won't exist and the mount is skipped.
_FRONTEND_DIR = Path(
    os.getenv("FRONTEND_DIR", str(Path(__file__).resolve().parent.parent / "frontend" / "build"))
)
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
