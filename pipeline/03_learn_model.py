import os 

import pandas as pd
import numpy as np
import s3fs
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


def plot_feature_importance(model_name, feature_imp, color_dict, top_n=10):
    plt.figure(figsize=(12, 6))
    
    top_features = feature_imp.nlargest(top_n, 'importance')
    
    bars = plt.bar(range(top_n), top_features['importance'], 
                   color=[color_dict[feat] for feat in top_features['feature']])
    
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=15, pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    
    plt.xticks(range(top_n), top_features['feature'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height(), 
                f'{top_features["importance"].iloc[i]:.3f}',
                ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(f"{model_name}_feature_importance.png", dpi=300, bbox_inches='tight')
    mlflow.log_artifact(f"{model_name}_feature_importance.png")
    plt.show()
    plt.close()
    

HOST = "http://mlflow.mlflow:5000"
EXPREIMENT_NAME = "Bike Rental Prediction"

os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'admin123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio.minio"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'admin1234567'

fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url":os.getenv("MLFLOW_S3_ENDPOINT_URL"),  # Minio 서버 주소
    },
    key=os.getenv("AWS_ACCESS_KEY_ID"),                       # Minio access key
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),                    # Minio secret key
    use_ssl=False,                          # HTTP 사용시 False   
)

parquet_files = fs.glob('dataset/weather_data_recent_spark_preprocessing/*.parquet')
df = pd.concat([pd.read_parquet(f, filesystem=fs) for f in parquet_files])

mlflow.set_tracking_uri(HOST)

X, y = df.drop('rental_cnt', axis=1), df["rental_cnt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 

                                                    random_state=42)

mlflow.set_tracking_uri(HOST)
mlflow.set_experiment(EXPREIMENT_NAME)
mlflow.sklearn.autolog()
mlflow.xgboost.autolog()
mlflow.enable_system_metrics_logging()

# 모델 목록
models = {
    "RandomForest100": RandomForestRegressor(random_state=42),
    "RandomForest200": RandomForestRegressor(n_estimators=200, random_state=42),
    "RandomForest300": RandomForestRegressor(n_estimators=300, random_state=42),
    "RandomForest350": RandomForestRegressor(n_estimators=350, random_state=42)
}

features = list(X_train.columns)
color_palette = sns.color_palette("husl", len(features))
color_dict = dict(zip(features, color_palette))
    
# 각 모델에 대해 학습 및 평가
for model_name, model in models.items():
    print("tryting to start learn model " + model_name)
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # MLflow에 메트릭 기록
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        
        # 모델 파라미터 기록
        mlflow.log_params(model.get_params())
        # mlflow.sklearn.log_model(sk_model=model, name=model_name)
        
        importances = None
        if model_name == "RandomForest":
            importances = model.feature_importances_
        elif model_name == "XGBoost":
            importances = model.feature_importances_
        elif model_name == "DecisionTree":
            importances = model.feature_importances_
        
        if importances is not None:
            feature_imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)

            feature_imp.to_csv(f"{model_name}_feature_importance.csv", index=False)
            mlflow.log_artifact(f"{model_name}_feature_importance.csv")

            plot_feature_importance(model_name, feature_imp, color_dict, top_n=len(features))
        
        print(f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

        
        
client = MlflowClient(tracking_uri=HOST)
experiment = client.get_experiment_by_name(EXPREIMENT_NAME)
experiment_id = experiment.experiment_id

runs = client.search_runs(experiment_id)
best_run = min(runs, key=lambda run: run.data.metrics['MSE'])

print(f"Best model: {best_run.data.tags['mlflow.runName']}")
print(f"MSE: {best_run.data.metrics['MSE']:.4f}")
print(f"RMSE: {best_run.data.metrics['RMSE']:.4f}")
print(f"R2: {best_run.data.metrics['R2']:.4f}")

model_name = "best_bike_rental_model"
model_uri = f"runs:/{best_run.info.run_id}/model"
try:
    new_model_version = mlflow.register_model(model_uri, model_name)
    print(f"New model version created: {new_model_version.version}")
except mlflow.exceptions.RestException:
    print(f"Model '{model_name}' already exists. Creating a new version.")
    new_model_version = mlflow.register_model(model_uri, model_name)
    print(f"New model version created: {new_model_version.version}")
