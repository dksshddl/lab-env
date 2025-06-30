import os
import socket

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

import s3fs
import pickle

sc = SparkSession.builder.appName("spark-preprocessing").getOrCreate()

# s3fs 파일시스템 객체 생성
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url':'http://minio.minio'},
    key='admin',
    secret='admin123',
    use_ssl=False
)


df = sc.read \
    .options(delimeter=',',header=True, inferSchema=True) \
    .csv("s3a://dataset/weather_data_recent_raw.csv")

df = df.drop('uvi')

visibility_mean = df.select(F.mean('visibility')).collect()[0][0]
df = df.na.fill({
    'visibility': visibility_mean
})


df = df.withColumn('isHoliday', 
    F.when(F.col('isHoliday') == False, "No Holiday")
     .when(F.col('isHoliday') == True, "Holiday")
)


df = df.withColumn('dt_timestamp', F.to_timestamp(F.col('dt')))
df = df.withColumn('hour diff', 
    (F.unix_timestamp('dt_timestamp') - F.unix_timestamp(F.to_timestamp(F.lit('2017-12-01')))) / 3600)


df_x = df.select(
    # Date와 Hour 컬럼 생성
    F.to_date('dt_timestamp').alias('Date'),
    F.hour('dt_timestamp').alias('Hour'),
    
    # 기존 컬럼들을 새로운 이름으로 매핑
    F.col('temp').alias('Temperature(C)'),
    F.col('humidity').alias('Humidity(%)'),
    F.col('wind_speed').alias('Wind speed (m/s)'),
    F.col('visibility').alias('Visibility (10m)'),
    F.col('dew_point').alias('Dew point temperature(C)'),
    
    # 비와 눈 데이터 변환
    F.col('rain').alias('Rainfall(mm)'),
    F.col('snow').alias('Snowfall (cm)'),
    
    # 계절 추가 (날짜 기반으로 계절 계산)
    F.when((F.month('datetime').isin(12, 1, 2)), 'Winter')
     .when((F.month('datetime').isin(3, 4, 5)), 'Spring')
     .when((F.month('datetime').isin(6, 7, 8)), 'Summer')
     .when((F.month('datetime').isin(9, 10, 11)), 'Fall')
     .alias('Seasons'),
    
    F.col('isHoliday').alias('Holiday'),
    F.col("hour diff"),
    # Functioning Day는 모두 'Yes'로 설정 (실제 상황에 맞게 수정 필요)
    F.lit('Yes').alias('Functioning Day')
)

with fs.open('dataset/label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 매핑 딕셔너리 생성
mappings = {
    'Seasons': dict(zip(encoders['Seasons'].classes_, encoders['Seasons'].transform(encoders['Seasons'].classes_))),
    'Holiday': dict(zip(encoders['Holiday'].classes_, encoders['Holiday'].transform(encoders['Holiday'].classes_))),
    'Functioning Day': dict(zip(encoders['Functioning Day'].classes_, encoders['Functioning Day'].transform(encoders['Functioning Day'].classes_)))
}

print(mappings)

mapping_expr = {
    col: F.create_map([F.lit(x) for x in sum(mapping.items(), ())]) 
    for col, mapping in mappings.items()
}

# 한 번에 모든 컬럼에 매핑 적용
for col, expr in mapping_expr.items():
    df_x = df_x.withColumn(col, expr.getItem(F.col(col)).cast('integer'))
    

    df_y = sc.read \
    .options(delimeter=',',header=True, inferSchema=True) \
    .csv("s3a://dataset/rental_bike_count.csv")

df_y = df_y.withColumn("datetime", F.to_timestamp(F.col("datetime").cast("string"), "yyyyMMdd"))


df_y = df_y.withColumn("date_hour", 
    F.concat(
        F.date_format("datetime", "yyyy-MM-dd"),
        F.lit(" "),
        F.lpad(F.col("hour").cast("string"), 2, "0")
    ))

df_x = df_x.withColumn("date_hour", 
    F.concat(
        F.col("Date"),
        F.lit(" "), 
        F.lpad(F.col("Hour").cast("string"), 2, "0")
    ))

merged_df = df_x.join(df_y, 
    (df_x.date_hour == df_y.date_hour),
    "left")


merged_df = merged_df.drop(df_y.hour, "Date", "date_hour", "datetime")


merged_df.write \
    .mode('overwrite') \
    .parquet('s3a://dataset/weather_data_recent_spark_preprocessing')