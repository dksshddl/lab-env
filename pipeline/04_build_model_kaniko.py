import base64
import string
import subprocess
import os

import boto3
from botocore.config import Config

from minio import Minio
import mlflow
from mlflow.tracking import MlflowClient

HOST = "http://mlflow.mlflow:5000"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "minio.minio"
os.environ['AWS_REGION'] = 'ap-northeast-2'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'admin1234567'
os.environ["MODEL_NAME"] = "seoulbike-rental-prediction-rf"
os.environ["MODEL_VERSION"] = "Version 1"
os.environ["CONTAINER_REGISTRY"] = "013596862746.dkr.ecr.ap-northeast-2.amazonaws.com"


model_name = os.environ["MODEL_NAME"]
model_version = os.environ["MODEL_VERSION"]
build_name = f"seldon-model-{model_name}-v{model_version}"

def init():
    mlflow.set_tracking_uri(HOST)
    

def create_ecr_repository(repository_name):
    print("create ecr repository if not exist")
    del os.environ["AWS_ACCESS_KEY_ID"]
    del os.environ["AWS_SECRET_ACCESS_KEY"]
    ecr_client = boto3.client('ecr')
    try:
        
        response = ecr_client.create_repository(
            repositoryName=repository_name,
            encryptionConfiguration={
                'encryptionType': 'AES256'
            }
        )
        print(f"Created repository {repository_name}")
        return repository_name
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository {repository_name} already exists")
        return repository_name
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def download_artifacts():
    print("retrieving model metadata from mlflow...")
    client = MlflowClient()
    bucket_name = "mlflow-artifact"
    
    model = client.get_registered_model(model_name)

    print(model)

    run_id = model.latest_versions[0].run_id
    source = model.latest_versions[0].source
    experiment_id = "1"
    
    print("initializing connection to s3 server...")
    minioClient = Minio(endpoint=os.environ['MLFLOW_S3_ENDPOINT_URL'],  # MinIO 서버 주소
                        access_key="admin",
                        secret_key="admin123",
                        secure=False)
    base_obj_path = source.split("mlflow-artifact")[1]
    data_file_model = minioClient.fget_object(bucket_name, f"{base_obj_path}/model.pkl", "model.pkl")
    data_file_requirements = minioClient.fget_object(bucket_name, f"/{base_obj_path}/requirements.txt", "requirements.txt")
    print("download successful")

    return run_id

    
def build_push_image(repo_name):
    container_image = repo_name
    container_tag = os.environ.get("CONTAINER_TAG", "latest")
    
    container_location = string.Template(f"$CONTAINER_REGISTRY/{container_image}:{container_tag}").substitute(os.environ)
        
    full_command = "/kaniko/executor --context=" + os.getcwd() + " --dockerfile=Predictor/Dockerfile --verbosity=debug --cache=true --single-snapshot=true --destination=" + container_location
    print(full_command)
    try:
        process = subprocess.run(full_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit status {e.returncode}")
        print(f"stderr: {e.stderr.decode()}")
    
    
# login_to_ecr()
repo_name = "mlops/basic-model"
init()
download_artifacts()
create_ecr_repository(repo_name)
build_push_image(repo_name)