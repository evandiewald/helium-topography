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
async def topography(request: Request, response: Response, address: str):
    result = get_results_for_hotspot(session, address)
    if result:
        return result
    else:
        return JSONResponse({"NoResultFound": "No topography result found for this address"}, status_code=500)


@router.get("/witnesses/{address}")
async def witnesses(request: Request, response: Response, address: str):
    sql_same_maker = f"""
    select

    rx_address as address,

    sum(
    	CASE WHEN
    	tx_payer = rx_payer THEN 1
    	ELSE 0
    	END
    )::float / count(tx_address) as same_maker_ratio, 

    count(*) as n_witnessed

    from detailed_receipts

    where rx_address = '{address}'
     group by rx_address;"""

    try:
        res = session.execute(sql_same_maker).one()
        print(res)
        return JSONResponse({"result": {"address": res[0], "different_maker_ratio": round(1 - res[1], 2), "n_witnessed": int(res[2])}})
    except sqlalchemy.exc.NoResultFound:
        return JSONResponse({"NoResultFound": "No witness result found for this address"}, status_code=500)



app.include_router(router)
