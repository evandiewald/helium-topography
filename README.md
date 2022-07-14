# helium-topography
Topographical modeling of the Helium Network with applications.

## Overview

This repository contains a variety of scripts that are in various stages of development. Overall, I am working toward refactoring the code away from ArangoDB and toward a place where all data is drawn from an instance of [`helium-transaction-etl`], a lightweight block follower and database. This will hopefully simplify deployments of the various scripts and applications. I'll try to keep this README up-to-date with the latest status of the various components.

| Script                | Description | Status                                             |
|:----------------------| ----------- |----------------------------------------------------|
| `api_batch.py`        | A REST API that serves topographic and witnessing metrics. | Stable. Used in [crowdspot](https://crowdspot.io). |
| `app.py`              | A streamlit app that displays trilateration and topography results **using ArangoDB as a backend**. | Functional, but no longer supported.               |
| `app_relational.py`   | Same as `app.py`, but using `helium-transaction-etl`, a SQL database. | Stable. (use this version of the app)              |
| `batch_processing.py` | Generates topographic predictions en masse and inserts results into the db populated by `helium-transaction-etl`. | Stable                                             |
| `train.py` | Trains the topographic ML models used by the above tools. Refactoring in progress. | Functional, no longer supported.                   |

## Dependencies (Ubuntu)

### Witness Data
These tools use a Postgres database that is populated with witness data via a helium block follower, [`blockchain-node`](https://github.com/helium/blockchain-node). 

1. Follow [these instructions](https://github.com/evandiewald/helium-transaction-etl) to run the `helium-transaction-etl` client alongside `blockchain-node`. Allow the service some time to ingest blocks.
2. Make a copy of `.env.template` called `.env` and populate the environment variables to link to your Postgres database and [Mapbox](https://www.mapbox.com/) API token.

## Elevation Maps, Trained Models, and Python Dependencies

The tools also draw from [open-source topographic datasets](https://portal.opentopography.org/raster?opentopoID=OTSRTM.042013.4326.1) courtesy of the Space Shuttle *Endeavor*. You'll need to download the entire map (~18GB) to your server. Further, we host pre-trained models that you can download if you want to skip the training step.

1. Create the following folders/subfolders within this repository directory.

```
mkdir -p static/gis-data/SRTM_GL3
mkdir -p static/trained_models/svm
mkdir -p static/trained_models/gaussian_process
mkdir -p static/trained_models/isolation_forest
```

2. Install the latest version of the AWS CLI. Instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
3. (from this directory) Download the SRTM dataset:

`aws s3 cp s3://raster/SRTM_GL3/ static/gis-data/SRTM_GL3 --recursive --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request`

4. Download the trained models:

```
wget -O static/trained_models/svm/2022-02-06T16_23_54.mdl https://helium-topography.s3.amazonaws.com/trained_models/svm/2022-02-04T16_31_09.mdl
wget -O static/trained_models/gaussian_process/2022-02-04T16_28_14.mdl https://helium-topography.s3.amazonaws.com/trained_models/gaussian_process/2022-02-04T16_28_14.mdl
wget -O static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl https://helium-topography.s3.amazonaws.com/trained_models/isolation_forest/2022-02-04T16_31_09.mdl
```

5. Initialize, activate, and install `requirements.txt` to a Python 3.7+ virtual environment

```shell
virtualenv venv
source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

You should now be able to run the scripts and apps mentioned above, e.g. 

`streamlit run app_relational.py` (launches the webapp on port 8501 by default)

`python batch_processing.py`