import aioredis

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from starlette.requests import Request
from starlette.responses import Response

import os
from pyArango.connection import Connection, Database
from app import get_hotspot_dict, find_outliers, monte_carlo_trilateration, generate_features, load_model, probability_by_hex_resolution
from dotenv import load_dotenv

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache


load_dotenv()

app = FastAPI()


TRAINED_SVM_PATH = "static/trained_models/svm/2022-02-06T16_23_54.mdl"
TRAINED_ISO_PATH = "static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl"

iso_model = load_model(TRAINED_ISO_PATH)
svm_model = load_model(TRAINED_SVM_PATH)

try:
    c = Connection(
        arangoURL=os.getenv('ARANGO_URL'),
        username=os.getenv('ARANGO_USERNAME'),
        password=os.getenv('ARANGO_PASSWORD')
    )
except ConnectionError:
    raise Exception('Unable to connect to the ArangoDB instance. Please check that it is running and that you have supplied the correct URL/credentials in the .env file.')
db: Database = c['helium-graphs']


@app.get("/trilateration/{address}")
@cache(expire=86400)
async def trilateration(request: Request, response: Response, address: str):
    try:
        hotspot_dict = get_hotspot_dict(db, address)
        features_df, details_df, witness_coords, _ = generate_features(db, address, "inbound")

        monte_carlo_results, _, _, prediction_error = monte_carlo_trilateration(features_df, witness_coords, svm_model, hotspot_dict, 5)

        p, _ = probability_by_hex_resolution(monte_carlo_results, hotspot_dict, 5)

        result = {
            "address": address,
            "prediction_error_km": prediction_error,
            "percent_predictions_within_5_krings": p
        }
        return result
    except:
        return {"error": "error processing request"}


@app.get("/anomalies/{address}")
@cache(expire=86400)
async def anomalies(request: Request, response: Response, address: str):
    try:
        hotspot_dict = get_hotspot_dict(db, address)
        features_df, details_df, witness_coords, _ = generate_features(db, address, "inbound")
        output_df = find_outliers(features_df, details_df, iso_model)

        _, _, _, prediction_error = monte_carlo_trilateration(features_df, witness_coords, svm_model, hotspot_dict, 1)

        result = {
            "address": address,
            "n_receipts": len(output_df),
            "n_outliers": len(output_df[output_df["classification"] < 0])
        }
        return result
    except:
        return {"error": "error processing request"}


@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")