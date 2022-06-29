import time

import sqlalchemy.exc

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

import connection
from models.tables import TopographyResults

from starlette.requests import Request
from starlette.responses import Response

import os
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

MAINTENANCE_MODE = True


router = APIRouter(prefix="/api/v1")
app = FastAPI()



if not MAINTENANCE_MODE:
    lite_engine = create_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
    lite_session = sessionmaker(lite_engine)

    etl_engine = connection.connect()
    etl_session = sessionmaker(etl_engine)
else:
    print("\n\nStarting server in MAINTENANCE MODE. Requests will be processed, but results will be empty.\n\n")


def get_results_for_hotspot(session: Session, address: str):
    stmt = select(TopographyResults).filter_by(address=address)
    try:
        res = session.execute(stmt).one()
        return res
    except sqlalchemy.exc.NoResultFound:
        return None


@router.get("/topography/{address}")
async def topography(request: Request, response: Response, address: str):
    if MAINTENANCE_MODE:
        return JSONResponse({"NoResultFound": "Systems under maintenance."}, status_code=500)
    else:
        with lite_session() as sess:
            result = get_results_for_hotspot(sess, address)
            if result:
                return result
            else:
                return JSONResponse({"NoResultFound": "No topography result found for this address"}, status_code=500)


@router.get("/witnesses/{address}")
async def witnesses(request: Request, response: Response, address: str):
    if MAINTENANCE_MODE:
        return JSONResponse({"NoResultFound": "Systems under maintenance."}, status_code=500)
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
