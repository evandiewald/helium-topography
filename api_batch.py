import aioredis
import sqlalchemy.exc

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from models.tables import TopographyResults

from starlette.requests import Request
from starlette.responses import Response

import os
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from dotenv import load_dotenv

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache


load_dotenv()

router = APIRouter(prefix="/api/v1")
app = FastAPI()


engine = create_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
session = Session(engine)


def get_results_for_hotspot(session: Session, address: str):
    stmt = select(TopographyResults).filter_by(address=address)
    try:
        res = session.execute(stmt).one()
        return res
    except sqlalchemy.exc.NoResultFound:
        return None


@router.get("/topography/{address}")
# @cache(expire=3600)
async def topography(request: Request, response: Response, address: str):
    result = get_results_for_hotspot(session, address)
    if result:
        return result
    else:
        return JSONResponse({"NoResultFound": "No topography result found for this address"}, status_code=500)

app.include_router(router)

# @app.on_event("startup")
# async def startup():
#     redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
#     FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")