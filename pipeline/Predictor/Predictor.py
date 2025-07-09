import os

import joblib
import mlflow


HOST = "http://mlflow.mlflow:5000"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio.minio"
os.environ['AWS_REGION'] = 'ap-northeast-2'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'admin1234567'
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["MODEL_NAME"] = "best_bike_rental_model"
os.environ["AWS_SECRET_ACCESS_KEY"] = "admin123"

class Predictor(object):

    def __init__(self):
        mlflow.set_tracking_uri(HOST)
        client = mlflow.MlflowClient()
        models = client.get_registered_model(os.environ["MODEL_NAME"])
        self.model = mlflow.pyfunc.load_model(models.latest_versions[0].source)

    def predict(self, data_array, column_names):
        return self.model.predict(data_array)