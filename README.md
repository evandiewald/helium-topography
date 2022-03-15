# helium-topography
Topographical modeling of the Helium Network with applications.

## Getting Started

This tool uses a graph database ([ArangoDB](https://www.arangodb.com/graph-database/)) that is populated with witness data via a helium block follower, [`blockchain-node`](https://github.com/helium/blockchain-node). 

1. Follow [these instructions](https://github.com/evandiewald/helium-arango-etl-lite) to run the `helium-arango-etl-lite` client alongside `blockchain-node`. Allow the service some time to ingest blocks.
2. Make a copy of `.env.template` called `.env` and populate the environment variables to link to your ArangoDB database and [Mapbox](https://www.mapbox.com/) API token.
3. Build the docker image with `docker build -t topo .`. It will take some time to download the [topographic dataset](https://portal.opentopography.org/raster?opentopoID=OTSRTM.042013.4326.1) (~18GB) and trained model parameters.
4. Run the container with `docker run -p 8501:8501 -d --name streamlit topo`.
5. If port 8501 is open, you should be able to access the streamlit app at `http://<HOSTNAME>:8501`.
