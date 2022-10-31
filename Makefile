docker-build-processing:
        docker build \
                -f Dockerfile.processing \
                -t helium-topography-batch-processing .

docker-clean-processing: docker-stop-processing
        docker rm batch-processing

docker-start-processing:
        docker run -d --init \
                --name batch-processing \
                helium-topography-batch-processing

docker-stop-processing:
        docker stop batch-processing

docker-build-api:
        docker build \
                -f Dockerfile.api \
                -t helium-topography-api .

docker-start-api:
        docker run -d --init \
                --name api \
                --publish 8080:8080 \
                helium-topography-api

docker-stop-api:
        docker stop api

docker-clean-api: docker-stop-api
        docker rm api
