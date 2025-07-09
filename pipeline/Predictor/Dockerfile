FROM python:3.11

RUN mkdir /microservice

WORKDIR /microservice

COPY Predictor/base_requirements.txt /microservice/
RUN pip install -r base_requirements.txt

# COPY requirements.txt /microservice/
# RUN pip install -r requirements.txt

COPY Predictor/Predictor.py /microservice/

CMD seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --grpc-port ${GRPC_PORT} --metrics-port ${METRICS_PORT} --http-port ${HTTP_PORT}
