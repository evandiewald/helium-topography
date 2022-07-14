FROM python:3.8-slim-buster

WORKDIR /usr/src/app

RUN mkdir -p static/trained_models/svm
RUN mkdir -p static/trained_models/gaussian_process
RUN mkdir -p static/trained_models/isolation_forest
RUN mkdir -p static/gis-data/SRTM_GL3

COPY . .

RUN apt-get upgrade
RUN apt-get -y update

RUN apt-get -y install unzip curl wget gcc

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN wget -O static/trained_models/svm/2022-02-06T16_23_54.mdl https://helium-topography.s3.amazonaws.com/trained_models/svm/2022-02-04T16_31_09.mdl
RUN wget -O static/trained_models/gaussian_process/2022-02-04T16_28_14.mdl https://helium-topography.s3.amazonaws.com/trained_models/gaussian_process/2022-02-04T16_28_14.mdl
RUN wget -O static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl https://helium-topography.s3.amazonaws.com/trained_models/isolation_forest/2022-02-04T16_31_09.mdl

# download SRTM dataset
RUN aws s3 cp s3://raster/SRTM_GL3/ static/gis-data/SRTM_GL3 --recursive --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install rasterio


CMD ["streamlit", "run", "app_relational.py"]