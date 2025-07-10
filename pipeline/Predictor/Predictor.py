import os

import joblib
import pandas as pd
import numpy as np
import mlflow


HOST = "http://mlflow.mlflow:5000"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio.minio"
os.environ['AWS_REGION'] = 'ap-northeast-2'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'admin1234567'
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["MODEL_NAME"] = "best_bike_rental_model"
os.environ["AWS_SECRET_ACCESS_KEY"] = "admin123"

mlflow_to_numpy_dtype = {
    "integer": "int32",
    "long": "int64",
    "double": "float64",
    "float": "float32",
    "boolean": "bool",
    "string": "object"
}


class Predictor(object):
    def __init__(self):
        mlflow.set_tracking_uri(HOST)
        client = mlflow.MlflowClient()
        models = client.get_registered_model(os.environ["MODEL_NAME"])
        self.model = mlflow.pyfunc.load_model(models.latest_versions[0].source)
        schema = self.model.metadata.get_input_schema()

        self.expected_columns = [col.name for col in schema]
        self.expected_types = {
            col.name: mlflow_to_numpy_dtype.get(str(col.type).split(".")[1], "object") for col in schema
        }

    def predict(self, X, features_names=None):
        if isinstance(X, list) or isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.expected_columns)
        elif isinstance(X, pd.DataFrame):
            X = X[self.expected_columns]
            
        # 명시적 dtype 캐스팅
        X = X.astype(self.expected_types)
        print(X.dtypes)
        return self.model.predict(X)
