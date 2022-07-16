import time

import sqlalchemy.exc

from fastapi import FastAPI, APIRouter, Request, Response
from fastapi.responses import JSONResponse

import connection
import models.tables
from models.tables import TopographyResults

from typing import Tuple
from cachetools import TTLCache, LRUCache

import os
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select
from dotenv import load_dotenv
import asyncio


load_dotenv()

MAINTENANCE_MODE = False
LITE_MODE = False
CACHE_ENABLED = False


router = APIRouter(prefix="/api/v1")
app = FastAPI()


if not MAINTENANCE_MODE:
    lite_engine = create_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
    lite_session = sessionmaker(lite_engine)
    print("Session initialized successfully.")

    if not LITE_MODE:
        etl_engine = connection.connect()
        etl_session = sessionmaker(etl_engine)
else:
    print("\n\nStarting server in MAINTENANCE MODE. Requests will be processed, but results will be empty.\n\n")


def get_topography_results(session: Session, address: str) -> Tuple[dict, int]:
    print("Getting topo results")
    stmt = select(TopographyResults).filter_by(address=address)
    try:
        res = session.execute(stmt).one()[0]
        result_dict = {
            "TopographyResults": {
                "percent_predictions_within_5_res8_krings": res.percent_predictions_within_5_res8_krings,
                "n_outliers": res.n_outliers,
                "block": res.block,
                "address": res.address,
                "prediction_error_km": res.prediction_error_km,
                "n_beaconers_heard": res.n_beaconers_heard
            }
        }
        return result_dict, 200
    except sqlalchemy.exc.NoResultFound:
        return {"NoResultFound": "No topography result found for this address"}, 500


def get_different_maker_ratio(session: Session, address: str) -> Tuple[dict, int]:
    print("getting same maker ratio")
    stmt = f"""select sum(case when tx_payer = rx_payer then 0 else 1 end)::float / count(*) as different_maker_ratio, count(*) as n_witnessed
        from detailed_receipts where rx_address = '{address}';"""
    result = session.execute(stmt).one()
    if result[0] is not None:
        return {"result": {"address": address, "different_maker_ratio": round(result[0], 2), "n_witnessed": int(result[1])}}, 200
    else:
        return {"NoResultFound": "No witness result found for this address"}, 500


# class WitnessCache(LRUCache):
#     def __missing__(self, address) -> asyncio.Task:
#         # Create a task
#         with lite_session() as session:
#             resource_future = asyncio.create_task(get_different_maker_ratio(session, address))
#         self[address] = resource_future
#         return resource_future


# class TopographyCache(LRUCache):
#     def __missing__(self, address) -> asyncio.Task:
#         # Create a task
#         with lite_session() as session:
#             resource_future = asyncio.create_task(get_topography_results(session, address))
#         self[address] = resource_future
#         return resource_future


# if CACHE_ENABLED:
#     witness_cache = WitnessCache(maxsize=1000)
#     topo_cache = TopographyCache(maxsize=1000)


@router.get("/topography/{address}")
async def topography(request: Request, response: Response, address: str):
    if MAINTENANCE_MODE:
        return JSONResponse({"NoResultFound": "Systems under maintenance."}, status_code=500)
    else:
        with lite_session() as session:
            result, status_code = get_topography_results(session, address)
        return JSONResponse(result, status_code=status_code)


@router.get("/witnesses/{address}")
async def witnesses(request: Request, response: Response, address: str):
    if MAINTENANCE_MODE:
        return JSONResponse({"NoResultFound": "Systems under maintenance."}, status_code=500)
    else:
        if LITE_MODE:
            with lite_session() as session:
                result, status_code = get_different_maker_ratio(session, address)
            return JSONResponse(result, status_code=status_code)
        else:
            sql_same_maker = f"""with gateway_details as (select last_block, payer from gateway_inventory where address = '{address}'),
            
            hashes as
            
            (select transaction_hash from transaction_actors 
            where actor_role = 'witness'::transaction_actor_role 
            and block > (select last_block from gateway_details) - 7500 
            and actor = '{address}'),
            
            metadata as
            (select text(fields->'path'->0->'challengee') as w from transactions where hash in (select * from hashes)),
            
            
            witnesses as
            (select substring(w from 2 for LENGTH(w)-2) as witness from metadata),
            
            
            
            makers as
            (select payer from witnesses b join gateway_inventory g on b.witness = g.address)
            
            
            select sum(case when payer = (select payer from gateway_details) then 1 else 0 end)::float / count(*) as same_maker_ratio, count(*) as n_witnessed from makers;"""

            with etl_session() as sess:
                try:
                    res = sess.execute(sql_same_maker).one()
                    return JSONResponse({"result": {"address": address, "different_maker_ratio": round(1 - res[0], 2), "n_witnessed": int(res[1])}})
                except sqlalchemy.exc.NoResultFound:
                    return JSONResponse({"NoResultFound": "No witness result found for this address"}, status_code=500)



app.include_router(router)
