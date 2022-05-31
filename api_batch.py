import sqlalchemy.exc

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

import connection
from models.tables import TopographyResults

from starlette.requests import Request
from starlette.responses import Response

import os
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from dotenv import load_dotenv


load_dotenv()

router = APIRouter(prefix="/api/v1")
app = FastAPI()


engine = create_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
session = Session(engine)

etl_engine = connection.connect()
etl_session = Session(etl_engine)



def get_results_for_hotspot(session: Session, address: str):
    stmt = select(TopographyResults).filter_by(address=address)
    try:
        res = session.execute(stmt).one()
        return res
    except sqlalchemy.exc.NoResultFound:
        return None


@router.get("/topography/{address}")
async def topography(request: Request, response: Response, address: str):
    result = get_results_for_hotspot(session, address)
    if result:
        return result
    else:
        return JSONResponse({"NoResultFound": "No topography result found for this address"}, status_code=500)


@router.get("/witnesses/{address}")
async def witnesses(request: Request, response: Response, address: str):
    sql_same_maker = f"""with gateway_details as (select last_block, payer from gateway_inventory where address = '{address}'),

    hashes as
    
    (select transaction_hash from transaction_actors 
    where actor_role = 'witness'::transaction_actor_role 
    and block > (select last_block from gateway_details) - 5000 
    and actor = '{address}'),
    
    metadata as
    
    (select fields->'path'->0->'witnesses' as w from transactions where hash in (select * from hashes)),
    
    results as
    (select
    (select array_agg(t -> 'gateway') from jsonb_array_elements(w) as x(t)) as gateway from metadata),
    
    witnesses_unnest as
    (select distinct(text(unnest(gateway))) as witness from results where gateway is not NULL),
    
    witnesses as
    (select substring(witness from 2 for LENGTH(witness)-2) as witness from witnesses_unnest),
    
    makers as
    (select payer from witnesses b join gateway_inventory g on b.witness = g.address)
    
    select sum(case when payer = (select payer from gateway_details) then 1 else 0 end)::float / count(*) as same_maker_ratio, count(*) as n_witnessed from makers;"""

    try:
        res = etl_session.execute(sql_same_maker).one()
        return JSONResponse({"result": {"address": address, "different_maker_ratio": round(1 - res[0], 2), "n_witnessed": int(res[1])}})
    except sqlalchemy.exc.NoResultFound:
        return JSONResponse({"NoResultFound": "No witness result found for this address"}, status_code=500)



app.include_router(router)
