import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor

def create_minio_bucket(minio_client: Minio, 
                        *bucket_names):
    # MinIO 클라이언트 설정
    minio_client = Minio(
        endpoint="minio.minio",  # MinIO 서버 주소
        access_key="admin",
        secret_key="admin123",
        secure=False
    )

    for bucket_name in bucket_names:
        try:
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
                print(f"MinIO bucket '{bucket_name}' created successfully")
            else:
                print(f"MinIO bucket '{bucket_name}' already exists")
        except S3Error as e:
            print(f"Error creating MinIO bucket {bucket_name}: {e}")


def upload_directory_to_minio(minio_client: Minio,
                              local_path: str,
                              bucket_name: str,
                              minio_path: str = ""):
    
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name}")

    # Convert path to absolute path
    local_path = os.path.abspath(local_path)
 
    if not os.path.exists(local_path):
        print(f"file or directory does not exist: {local_path}")
        return

    if os.path.isfile(local_path):
        file_path = local_path.split("/")[-1]
        if minio_path:
            object_name = os.path.join(minio_path, file_path)
        else:
            object_name = file_path
        try:
            minio_client.fput_object(
                bucket_name,
                object_name,
                local_path
            )
            print(f"Uploaded {file_path} to {bucket_name}/{object_name}")
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
        return
    
    for root, dirs, files in os.walk(local_path):
        if ".ipynb_checkpoints" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, local_path)
            if minio_path:
                object_name = os.path.join(minio_path, relative_path)
            else:
                object_name = relative_path
            
            try:
                minio_client.fput_object(
                    bucket_name,
                    object_name,
                    file_path
                )
                print(f"Uploaded {file_path} to {bucket_name}/{object_name}")
            except Exception as e:
                print(f"Error uploading {file_path}: {str(e)}")
                


def preprocessing(file_name):
    df = pd.read_csv(file_name)

    df = df.drop('uvi', axis=1)

    visibility_mean = df['visibility'].mean()
    df['visibility'].fillna(visibility_mean, inplace=True)

    df['isHoliday'] = df['isHoliday'].map({False: "No Holiday", True: "Holiday"})

    print(f"num of null visibility: {df['visibility'].isnull().sum()}")

    df['dt_timestamp'] = pd.to_datetime(df['dt'], unit='s')
    base_date = pd.Timestamp('2017-12-01')
    df['hour diff'] = (df['dt_timestamp'] - base_date).dt.total_seconds() / 3600

    df_x = pd.DataFrame()
    df_x['Date'] = df['dt_timestamp'].dt.date
    df_x['Hour'] = df['dt_timestamp'].dt.hour
    df_x['Temperature(C)'] = df['temp']
    df_x['Humidity(%)'] = df['humidity']
    df_x['Wind speed (m/s)'] = df['wind_speed']
    df_x['Visibility (10m)'] = df['visibility']
    df_x['Dew point temperature(C)'] = df['dew_point']
    df_x['Rainfall(mm)'] = df['rain']
    df_x['Snowfall (cm)'] = df['snow']

    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    df_x['Seasons'] = df['dt_timestamp'].apply(get_season)
    df_x['Holiday'] = df['isHoliday']
    df_x['hour diff'] = df['hour diff']
    df_x['Functioning Day'] = 'Yes'

    import pickle
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    # 카테고리형 변수 인코딩
    for col in ['Seasons', 'Holiday', 'Functioning Day']:
        mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
        df_x[col] = df_x[col].map(mapping)

    # 대여 데이터 읽기
    df_y = pd.read_csv("dataset/rental_bike_count.csv")
    df_y['datetime'] = pd.to_datetime(df_y['datetime'].astype(str), format='%Y%m%d')

    # date_hour 컬럼 생성
    df_y['date_hour'] = df_y['datetime'].dt.strftime('%Y-%m-%d') + ' ' + df_y['hour'].astype(str).str.zfill(2)
    df_x['date_hour'] = df_x['Date'].astype(str) + ' ' + df_x['Hour'].astype(str).str.zfill(2)

    merged_df = pd.merge(df_x, df_y, on='date_hour', how='left')

    merged_df = merged_df.drop(['hour', 'Date', 'date_hour', 'datetime'], axis=1).to_csv("dataset/merged.csv", index=False)

    
def new_preprocessing(file_name):
    df = pd.read_csv(file_name)

    df = df.drop('uvi', axis=1)
    visibility_mean = df['visibility'].mean()
    df['visibility'].fillna(visibility_mean, inplace=True)

    df['isHoliday'] = df['isHoliday'].map({False: "No Holiday", True: "Holiday"})

    print(f"num of null visibility: {df['visibility'].isnull().sum()}")

    df['dt_timestamp'] = pd.to_datetime(df['dt'], unit='s')
    base_date = pd.Timestamp('2017-12-01')
    df['hour diff'] = (df['dt_timestamp'] - base_date).dt.total_seconds() / 3600

    df_x = pd.DataFrame()
    df_x['Date'] = df['dt_timestamp'].dt.date
    df_x['Hour'] = df['dt_timestamp'].dt.hour
    df_x['Temperature(C)'] = df['temp']
    df_x['Humidity(%)'] = df['humidity']
    df_x['Wind speed (m/s)'] = df['wind_speed']
    df_x['Visibility (10m)'] = df['visibility']
    df_x['Dew point temperature(C)'] = df['dew_point']
    df_x['Rainfall(mm)'] = df['rain']
    df_x['Snowfall (cm)'] = df['snow']
    df_x['weather'] = df['weather']

    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    # df_x['Seasons'] = df['dt_timestamp'].apply(get_season)
    df_x['Holiday'] = df['isHoliday']
    df_x['hour diff'] = df['hour diff']
    df_x['Functioning Day'] = 'Yes'

    numeric_features = [col for col in df_x.columns if col != 'weather' and col != 'target_column']
    categorical_features = ['weather']
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])
    X_encoded = preprocessor.fit_transform(df_x)
    feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['weather'])
    print(feature_namefeature_names)
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

    
    import pickle
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    # 카테고리형 변수 인코딩
    for col in ['Holiday', 'Functioning Day']:
        mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
        df_x[col] = df_x[col].map(mapping)

    # 대여 데이터 읽기
    df_y = pd.read_csv("dataset/rental_bike_count.csv")
    df_y['datetime'] = pd.to_datetime(df_y['datetime'].astype(str), format='%Y%m%d')

    # date_hour 컬럼 생성
    df_y['date_hour'] = df_y['datetime'].dt.strftime('%Y-%m-%d') + ' ' + df_y['hour'].astype(str).str.zfill(2)
    df_x['date_hour'] = df_x['Date'].astype(str) + ' ' + df_x['Hour'].astype(str).str.zfill(2)

    merged_df = pd.merge(X_encoded_df, df_y, on='date_hour', how='left')

    merged_df = merged_df.drop(['hour', 'Date', 'date_hour', 'datetime'], axis=1).to_csv("dataset/merged.csv")

    print(merged_df.head())
