import warnings

warnings.filterwarnings("ignore")

import os
import gc
import pickle
import datetime
import copy
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px

from sklearn.ensemble import VotingRegressor
import lightgbm as lgb

import holidays
import enefit


class DataStorage:
    root = "/kaggle/input/predict-energy-behavior-of-prosumers"

    data_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
    ]
    client_cols = [
        "product_type",
        "county",
        "eic_count", 
        "installed_capacity",
        "is_business",
        "date",
    ]
    gas_prices_cols = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
    electricity_prices_cols = ["forecast_date", "euros_per_mwh"]
    forecast_weather_cols = [
        "latitude",
        "longitude",
        "origin_datetime",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low", 
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    historical_weather_cols = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
    ]
    location_cols = ["longitude", "latitude", "county"]
    target_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
    ]
    
    def __init__(self):
        # 读取各个csv文件到polars DataFrame
        self.df_data = pl.read_csv(
            os.path.join(self.root, "train.csv"),
            columns=self.data_cols,
            try_parse_dates=True,
        )
        self.df_client = pl.read_csv(
            os.path.join(self.root, "client.csv"),
            columns=self.client_cols,
            try_parse_dates=True,  
        )
        self.df_gas_prices = pl.read_csv(
            os.path.join(self.root, "gas_prices.csv"),
            columns=self.gas_prices_cols,
            try_parse_dates=True,
        )
        self.df_electricity_prices = pl.read_csv(
            os.path.join(self.root, "electricity_prices.csv"),
            columns=self.electricity_prices_cols,
            try_parse_dates=True,
        )
        self.df_forecast_weather = pl.read_csv(
            os.path.join(self.root, "forecast_weather.csv"),
            columns=self.forecast_weather_cols,
            try_parse_dates=True,
        )
        self.df_historical_weather = pl.read_csv(
            os.path.join(self.root, "historical_weather.csv"),
            columns=self.historical_weather_cols, 
            try_parse_dates=True,
        )
        self.df_weather_station_to_county_mapping = pl.read_csv(
            os.path.join(self.root, "weather_station_to_county_mapping.csv"),
            columns=self.location_cols,
            try_parse_dates=True,
        )
        # 只保留2021年之后的训练数据
        self.df_data = self.df_data.filter(
            pl.col("datetime") >= pd.to_datetime("2021-01-01")
        )
        # 提取目标列
        self.df_target = self.df_data.select(self.target_cols)
        
        # 保存每个DataFrame的schema,方便后续更新数据时使用
        self.schema_data = self.df_data.schema
        self.schema_client = self.df_client.schema  
        self.schema_gas_prices = self.df_gas_prices.schema
        self.schema_electricity_prices = self.df_electricity_prices.schema
        self.schema_forecast_weather = self.df_forecast_weather.schema
        self.schema_historical_weather = self.df_historical_weather.schema
        self.schema_target = self.df_target.schema

        # 将经度纬度列转为float32类型  
        self.df_weather_station_to_county_mapping = (
            self.df_weather_station_to_county_mapping.with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32), 
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
        )
    
    # 使用新数据更新现有DataFrame  
    def update_with_new_data(
        self,
        df_new_client,  
        df_new_gas_prices,
        df_new_electricity_prices,
        df_new_forecast_weather,
        df_new_historical_weather,
        df_new_target,
    ):
        # 读取新数据到polars DataFrame,使用相应的schema
        df_new_client = pl.from_pandas(
            df_new_client[self.client_cols], schema_overrides=self.schema_client
        )
        df_new_gas_prices = pl.from_pandas(
            df_new_gas_prices[self.gas_prices_cols],
            schema_overrides=self.schema_gas_prices,
        )
        df_new_electricity_prices = pl.from_pandas( 
            df_new_electricity_prices[self.electricity_prices_cols],
            schema_overrides=self.schema_electricity_prices,
        )
        df_new_forecast_weather = pl.from_pandas(
            df_new_forecast_weather[self.forecast_weather_cols],
            schema_overrides=self.schema_forecast_weather,
        )
        df_new_historical_weather = pl.from_pandas(
            df_new_historical_weather[self.historical_weather_cols], 
            schema_overrides=self.schema_historical_weather,
        )
        df_new_target = pl.from_pandas(
            df_new_target[self.target_cols], schema_overrides=self.schema_target
        )

        # 把新旧数据合并,并保留唯一行
        self.df_client = pl.concat([self.df_client, df_new_client]).unique(
            ["date", "county", "is_business", "product_type"]
        )
        self.df_gas_prices = pl.concat([self.df_gas_prices, df_new_gas_prices]).unique(
            ["forecast_date"]
        )
        self.df_electricity_prices = pl.concat(
            [self.df_electricity_prices, df_new_electricity_prices]  
        ).unique(["forecast_date"])
        self.df_forecast_weather = pl.concat(
            [self.df_forecast_weather, df_new_forecast_weather]
        ).unique(["forecast_datetime", "latitude", "longitude", "hours_ahead"])
        self.df_historical_weather = pl.concat(
            [self.df_historical_weather, df_new_historical_weather]
        ).unique(["datetime", "latitude", "longitude"])
        self.df_target = pl.concat([self.df_target, df_new_target]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]   
        )
    
    # 预处理测试集DataFrame
    def preprocess_test(self, df_test):
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        df_test = pl.from_pandas(
            df_test[self.data_cols[1:]], schema_overrides=self.schema_data  
        )
        return df_test


class FeaturesGenerator:
    def __init__(self, data_storage):
        self.data_storage = data_storage
        # 读取爱沙尼亚2021-2025年的节假日列表
        self.estonian_holidays = list(
            holidays.country_holidays("EE", years=range(2021, 2026)).keys()  
        )
    
    # 添加一般特征
    def _add_general_features(self, df_features):
        df_features = (
            df_features.with_columns(
                pl.col("datetime").dt.ordinal_day().alias("dayofyear"), # 一年中的第几天
                pl.col("datetime").dt.hour().alias("hour"), # 小时
                pl.col("datetime").dt.day().alias("day"), # 日 
                pl.col("datetime").dt.weekday().alias("weekday"), # 周几
                pl.col("datetime").dt.month().alias("month"), # 月  
                pl.col("datetime").dt.year().alias("year"), # 年
            )
            .with_columns(
                pl.concat_str(
                    "county",  
                    "is_business",
                    "product_type",
                    "is_consumption", 
                    separator="_",
                ).alias("segment"),
            )
            .with_columns(
                (np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"), # 将一年中的第几天转换为正弦特征
                (np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"), # 余弦特征  
                (np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"), # 小时的正弦特征
                (np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"), # 余弦特征
            )
        )
        return df_features
    
    # 添加客户信息特征
    def _add_client_features(self, df_features):  
        df_client = self.data_storage.df_client

        df_features = df_features.join(
            df_client.with_columns(
                (pl.col("date") + pl.duration(days=2)).cast(pl.Date) # 客户表的日期往后移2天以匹配目标日期
            ),  
            on=["county", "is_business", "product_type", "date"], # 按县、是否商业客户、产品类型、日期合并
            how="left",
        )
        return df_features
        
    # 检查给定日期是否为爱沙尼亚节假日
    def is_country_holiday(self, row):
        return (  
            datetime.date(row["year"], row["month"], row["day"])
            in self.estonian_holidays
        )
    
    # 添加节日特征
    def _add_holidays_features(self, df_features):
        df_features = df_features.with_columns( 
            pl.struct(["year", "month", "day"])
            .apply(self.is_country_holiday)
            .alias("is_country_holiday")
        )
        return df_features
    
    def _add_forecast_weather_features(self, df_features): 
        # 获取预测天气数据和天气站与县的映射关系数据
        df_forecast_weather = self.data_storage.df_forecast_weather
        df_weather_station_to_county_mapping = (
            self.data_storage.df_weather_station_to_county_mapping 
        )

        # 处理预测天气数据
        df_forecast_weather = (
            df_forecast_weather.rename({"forecast_datetime": "datetime"}) # 将'forecast_datetime'重命名为'datetime'
            .filter((pl.col("hours_ahead") >= 22) & pl.col("hours_ahead") <= 45) # 只保留22-45小时的预报数据
            .drop("hours_ahead") # 删除'hours_ahead'列
            .with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32), # 将'latitude'列转换为Float32类型
                pl.col("longitude").cast(pl.datatypes.Float32), # 将'longitude'列转换为Float32类型
                pl.col("datetime").dt.hour().cast(pl.datatypes.Int64).alias("hour"), # 提取'datetime'中的小时数
                pl.col("datetime").dt.month().cast(pl.datatypes.Int64).alias("month"), # 提取'datetime'中的月份
            )
            .join(
                df_weather_station_to_county_mapping, # 与天气站到县的映射关系数据合并
                how="left",  
                on=["longitude", "latitude"], # 按经纬度合并
            )
            .drop("longitude", "latitude", "origin_datetime", "month", "hour") # 删除不需要的列
        )
        
        # 按预报时间分组,计算各特征的平均值,得到全局预报特征
        df_forecast_weather_date = (
            df_forecast_weather.group_by("datetime").mean().drop("county")
        )
        
        # 按预报时间和县分组,计算各特征的平均值,得到各县的预报特征
        df_forecast_weather_local = (
            df_forecast_weather.filter(pl.col("county").is_not_null())
            .group_by("county", "datetime")
            .mean()  
        )
        
        # 将全局和各县的预报特征按不同滞后期合并到原始特征中
        for hours_lag in [-1, 0, 1, 2 * 24, 7 * 24]:
            # 将全局天气特征滞后hours_lag小时的数据添加到特征中，并加上后缀_forecast_{hours_lag}h
            df_features = df_features.join(
                df_forecast_weather_date.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag) 
                ),
                on="datetime", 
                how="left",
                suffix=f"_forecast_{hours_lag}h", 
            )
            # 将各县天气特征滞后hours_lag小时的数据添加到特征中，并加上后缀_forecast_local_{hours_lag}h
            df_features = df_features.join(
                df_forecast_weather_local.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),  
                on=["county", "datetime"],
                how="left", 
                suffix=f"_forecast_local_{hours_lag}h", 
            )
            
        # 添加一些时间差特征
        df_features = df_features.with_columns(
            # 将当前时刻与168小时后的温度比值作为特征
            (
                pl.col(f"temperature_forecast_local_0h")
                / (pl.col(f"temperature_forecast_local_168h") + 1e-3)
            ).alias(f"temperature_forecast_local_0h/168h"), 

            # 将当前时刻与168小时后的向下太阳辐射比值作为特征
            (
                pl.col(f"surface_solar_radiation_downwards_forecast_local_0h")
                / (pl.col(f"surface_solar_radiation_downwards_forecast_local_168h") + 1e-3)
            ).alias(f"surface_solar_radiation_downwards_forecast_local_0h/168h"), 
        )
        
        # 添加移动平均特征

        # 计算前后1小时内的直接太阳辐射的平均值
        cols_for_stats = [f"direct_solar_radiation_forecast_local_{hours_lag}h" for hours_lag in [-1, 0, 1]]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"MA_direct_solar_radiation_forecast_local"), 
        )
        
        # 计算前后1小时内的向下太阳辐射的平均值
        cols_for_stats = [f"surface_solar_radiation_downwards_forecast_local_{hours_lag}h" for hours_lag in [-1, 0, 1]]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"MA_surface_solar_radiation_downwards_forecast_local"), 
        )
        
        # 计算前后1小时内的温度的平均值
        cols_for_stats = [f"temperature_forecast_local_{hours_lag}h" for hours_lag in [-1, 0, 1]]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"MA_temperature_forecast_local"), 
        )
        
        # 计算前后1小时内的总降水量的平均值
        cols_for_stats = [f"total_precipitation_forecast_local_{hours_lag}h" for hours_lag in [-1, 0, 1]]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"MA_total_precipitation_forecast_local"), 
        )

        return df_features


    def _add_target_features(self, df_features):
        df_target = self.data_storage.df_target

        # 按时间、县、是否商业客户、是否消费分组,计算target的和,得到product_type层面的聚合target
        df_target_all_type_sum = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .sum()
            .drop("product_type") 
        )

        # 按时间、是否商业客户、是否消费分组,计算target的和,得到county层面的聚合target
        df_target_all_county_type_sum = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .sum()
            .drop("product_type", "county")
        )

        # 添加更多target滞后特征 
        for hours_lag in [
            2 * 24, 2 * 24 - 1, 2 * 24 + 1, 
            3 * 24,
            4 * 24,
            5 * 24,
            6 * 24,
            7 * 24, 7 * 24 - 1, 7 * 24 + 1,
            8 * 24,
            9 * 24,
            10 * 24,
            11 * 24,
            12 * 24,
            13 * 24,
            14 * 24, 14 * 24 - 1, 14 * 24 + 1
        ]:
            # 为每个滞后时间添加target特征
            df_features = df_features.join(
                df_target.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename({"target": f"target_{hours_lag}h"}),
                on=[
                    "county",
                    "is_business",
                    "product_type",
                    "is_consumption",
                    "datetime",
                ],
                how="left",
            )

        # 添加product_type和county层面的聚合target特征
        for hours_lag in [2 * 24, 3 * 24, 7 * 24, 14 * 24]:
            # 按时间、县、是否商业客户、是否消费分组,添加滞后2天、3天、7天和14天的聚合target特征
            df_features = df_features.join(
                df_target_all_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename({"target": f"target_all_type_sum_{hours_lag}h"}), 
                on=["county", "is_business", "is_consumption", "datetime"],
                how="left",
            )
            
            # 按时间、是否商业客户、是否消费分组,添加滞后2天、3天、7天和14天的county层面的聚合target特征
            df_features = df_features.join(
                df_target_all_county_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename({"target": f"target_all_county_type_sum_{hours_lag}h"}),
                on=["is_business", "is_consumption", "datetime"],
                how="left",
                suffix=f"_all_county_type_sum_{hours_lag}h",
            )

        # 计算2-5天滞后target的统计特征
        cols_for_stats = [
            f"target_{hours_lag}h" for hours_lag in [2 * 24, 3 * 24, 4 * 24, 5 * 24]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"target_mean"), # 计算滞后2-5天target的均值
            df_features.select(cols_for_stats).max(axis=1).alias(f"target_max"), # 计算滞后2-5天target的最大值
            df_features.select(cols_for_stats).min(axis=1).alias(f"target_min"), # 计算滞后2-5天target的最小值
            df_features.select(cols_for_stats)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std"), # 计算滞后2-5天target的标准差
        )
        
        #===add===
        # 计算2-7天滞后target的统计特征
        cols_for_stats_2 = [
            f"target_{hours_lag}h" for hours_lag in [2 * 24, 3 * 24, 4 * 24, 5 * 24, 6 * 24, 7 * 24]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats_2).mean(axis=1).alias(f"target_mean_2"), # 计算滞后2-7天target的均值
            df_features.select(cols_for_stats_2).max(axis=1).alias(f"target_max_2"), # 计算滞后2-7天target的最大值
            df_features.select(cols_for_stats_2).min(axis=1).alias(f"target_min_2"), # 计算滞后2-7天target的最小值
            df_features.select(cols_for_stats_2)
            .transpose() 
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std_2"),  # 计算滞后2-7天target的标准差
        )
        
        # 计算7天和14天滞后target的统计特征
        cols_for_stats_3 = [
            f"target_{hours_lag}h" for hours_lag in [7 * 24, 14 * 24]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats_3).mean(axis=1).alias(f"target_mean_3"), # 计算滞后7-14天target的均值
            df_features.select(cols_for_stats_3).max(axis=1).alias(f"target_max_3"), # 计算滞后7-14天target的最大值
            df_features.select(cols_for_stats_3).min(axis=1).alias(f"target_min_3"), # 计算滞后7-14天target的最小值
            df_features.select(cols_for_stats_3)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std_3"), # 计算滞后7-14天target的标准差
        )
        
        # 计算2-14天滞后target的统计特征
        cols_for_stats_4 = [
            f"target_{hours_lag}h" for hours_lag in [2 * 24, 2 * 24 - 1, 2 * 24 + 1]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats_4).mean(axis=1).alias(f"target_mean_4"), # 计算滞后2-14天target的均值
            df_features.select(cols_for_stats_4).max(axis=1).alias(f"target_max_4"), # 计算滞后2-14天target的最大值
            df_features.select(cols_for_stats_4).min(axis=1).alias(f"target_min_4"), # 计算滞后2-14天target的最小值
            df_features.select(cols_for_stats_4)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std_4"), # 计算滞后2-14天target的标准差
        ) 
        
        # 计算7-14天滞后target的统计特征
        cols_for_stats_5 = [
            f"target_{hours_lag}h" for hours_lag in [7 * 24, 7 * 24 - 1, 7 * 24 + 1]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats_5).mean(axis=1).alias(f"target_mean_5"), # 计算滞后7-14天target的均值
            df_features.select(cols_for_stats_5).max(axis=1).alias(f"target_max_5"), # 计算滞后7-14天target的最大值
            df_features.select(cols_for_stats_5).min(axis=1).alias(f"target_min_5"), # 计算滞后7-14天target的最小值
            df_features.select(cols_for_stats_5)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std_5"), # 计算滞后7-14天target的标准差
        )
        
        # 计算14-21天滞后target的统计特征
        cols_for_stats_6 = [
            f"target_{hours_lag}h" for hours_lag in [14 * 24, 14 * 24 - 1, 14 * 24 + 1]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats_6).mean(axis=1).alias(f"target_mean_6"), # 计算滞后14-21天target的均值
            df_features.select(cols_for_stats_6).max(axis=1).alias(f"target_max_6"), # 计算滞后14-21天target的最大值
            df_features.select(cols_for_stats_6).min(axis=1).alias(f"target_min_6"), # 计算滞后14-21天target的最小值
            df_features.select(cols_for_stats_6)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std_6"), # 计算滞后14-21天target的标准差
        )

        # 计算2-3天滞后target的比率特征
        for target_prefix, lag_nominator, lag_denomonator in [
            ("target", 24 * 7, 24 * 14), # 滞后7天与滞后14天target的比率
            ("target", 24 * 2, 24 * 9), # 滞后2天与滞后9天target的比率
            ("target", 24 * 3, 24 * 10), # 滞后3天与滞后10天target的比率
            ("target", 24 * 2, 24 * 3), # 滞后2天与滞后3天target的比率
            ("target_all_type_sum", 24 * 2, 24 * 3), # 滞后2天与滞后3天的product_type层面的聚合target的比率
            ("target_all_type_sum", 24 * 7, 24 * 14), # 滞后7天与滞后14天的product_type层面的聚合target的比率
            ("target_all_county_type_sum", 24 * 2, 24 * 3), # 滞后2天与滞后3天的county层面的聚合target的比率
            ("target_all_county_type_sum", 24 * 7, 24 * 14), # 滞后7天与滞后14天的county层面的聚合target的比率
        ]:
            df_features = df_features.with_columns(
                (
                    pl.col(f"{target_prefix}_{lag_nominator}h")
                    / (pl.col(f"{target_prefix}_{lag_denomonator}h") + 1e-3)
                ).alias(f"{target_prefix}_ratio_{lag_nominator}_{lag_denomonator}")
            )

        return df_features

    def _reduce_memory_usage(self, df_features):
        # 将所有float64类型的列转为float32以减少内存占用
        df_features = df_features.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return df_features

    def _drop_columns(self, df_features):
        # 删除不需要的列
        df_features = df_features.drop(
            "date", "datetime", "hour", "dayofyear" 
        )  
        return df_features

    def _to_pandas(self, df_features, y):
        # 分类特征列
        cat_cols = [
            "county",    
            "is_business",
            "product_type",
            "is_consumption", 
            "segment",
        ]

        # 如果y不为None,则将df_features和y合并为一个pandas DataFrame
        if y is not None:
            df_features = pd.concat([df_features.to_pandas(), y.to_pandas()], axis=1)
        else:
            df_features = df_features.to_pandas()

        # 将分类特征转为category类型
        df_features[cat_cols] = df_features[cat_cols].astype("category")
        
        # 如果存在row_id列,则删除
        if 'row_id' in df_features.columns:
            df_features = df_features.drop("row_id", axis=1)

        return df_features

    def generate_features(self, df_prediction_items):
        # 如果df_prediction_items包含target列,则分离出target和其他特征  
        if "target" in df_prediction_items.columns:
            df_prediction_items, y = (
                df_prediction_items.drop("target"),
                df_prediction_items.select("target"),
            )
        else:
            y = None

        # 添加date列
        df_features = df_prediction_items.with_columns(
            pl.col("datetime").cast(pl.Date).alias("date"),
        )

        # 依次调用各个特征工程方法
        for add_features in [
            self._add_general_features,   # 添加一般特征
            self._add_client_features,    # 添加客户信息特征
            self._add_forecast_weather_features,  # 添加天气预报特征
            self._add_target_features,    # 添加目标值滞后特征
            self._add_holidays_features,  # 添加节假日特征
            self._reduce_memory_usage,    # 减少内存使用
            self._drop_columns,           # 删除不需要的列
        ]:
            df_features = add_features(df_features)

        # 添加一些额外的时间差特征
        df_features = (
            df_features.with_columns(
                (pl.col("direct_solar_radiation_forecast_local_0h") - pl.col("direct_solar_radiation_forecast_local_48h")).alias("diff_direct_solar_radiation_local_0h_48h"),  # 当前时刻与48小时前的直接太阳辐射差值
                (pl.col("direct_solar_radiation_forecast_local_0h") - pl.col("direct_solar_radiation_forecast_local_168h")).alias("diff_direct_solar_radiation_local_0h_168h"),  # 当前时刻与168小时前的直接太阳辐射差值
                (pl.col("direct_solar_radiation") - pl.col("direct_solar_radiation_forecast_48h")).alias("diff_direct_solar_radiation_0h_48h"),  # 当前时刻与48小时前的直接太阳辐射差值
                (pl.col("direct_solar_radiation") - pl.col("direct_solar_radiation_forecast_168h")).alias("diff_direct_solar_radiation_0h_168h"),  # 当前时刻与168小时前的直接太阳辐射差值
                (pl.col("temperature_forecast_local_0h") - pl.col("temperature_forecast_local_48h")).alias("diff_temperature_local_0h_48h"),  # 当前时刻与48小时前的温度差值
                (pl.col("temperature_forecast_local_0h") - pl.col("temperature_forecast_local_168h")).alias("diff_temperature_local_0h_168h"),  # 当前时刻与168小时前的温度差值
                (pl.col("temperature") - pl.col("temperature_forecast_48h")).alias("diff_temperature_0h_48h"),  # 当前时刻与48小时前的温度差值
                (pl.col("temperature") - pl.col("temperature_forecast_168h")).alias("diff_temperature_0h_168h"),  # 当前时刻与168小时前的温度差值

                (pl.col("direct_solar_radiation_forecast_local_0h")/(pl.col("temperature_forecast_local_0h")+273.15)).alias("dsr_temp_local_0h"),  # 当前时刻直接太阳辐射与温度的比值
                (pl.col("direct_solar_radiation")/(pl.col("temperature")+273.15)).alias("dsr_temp_0h"),  # 当前时刻直接太阳辐射与温度的比值
                (pl.col("direct_solar_radiation_forecast_local_48h")/(pl.col("temperature_forecast_local_48h")+273.15)).alias("dsr_temp_local_48h"),  # 48小时前直接太阳辐射与温度的比值
                (pl.col("direct_solar_radiation_forecast_48h")/(pl.col("temperature_forecast_48h")+273.15)).alias("dsr_temp_48h"),  # 48小时前直接太阳辐射与温度的比值
                (pl.col("direct_solar_radiation_forecast_local_168h")/(pl.col("temperature_forecast_local_168h")+273.15)).alias("dsr_temp_local_168h"),  # 168小时前直接太阳辐射与温度的比值
                (pl.col("direct_solar_radiation_forecast_168h")/(pl.col("temperature_forecast_168h")+273.15)).alias("dsr_temp_168h"),  # 168小时前直接太阳辐射与温度的比值
                
                (pl.col("MA_direct_solar_radiation_forecast_local")/(pl.col("MA_temperature_forecast_local")+273.15)).alias("MA_dsr_temp_local"),  # 移动平均的直接太阳辐射与温度的比值
            )
            .with_columns(
                (pl.col("dsr_temp_local_0h") - pl.col("dsr_temp_local_48h")).alias("diff_dsr_temp_local_0h_48h"),  # 当前时刻与48小时前的直接太阳辐射与温度比值的差值
                (pl.col("dsr_temp_local_0h") - pl.col("dsr_temp_local_168h")).alias("diff_dsr_temp_local_0h_168h"),  # 当前时刻与168小时前的直接太阳辐射与温度比值的差值
                (pl.col("dsr_temp_0h") - pl.col("dsr_temp_48h")).alias("diff_dsr_temp_0h_48h"),  # 当前时刻与48小时前的直接太阳辐射与温度比值的差值
                (pl.col("dsr_temp_0h") - pl.col("dsr_temp_168h")).alias("diff_dsr_temp_0h_168h"),  # 当前时刻与168小时前的直接太阳辐射与温度比值的差值
                
                (pl.col("dsr_temp_local_0h") * pl.col("installed_capacity")).alias("ie_dsr_temp_local_0h"),  # 当前时刻直接太阳辐射与温度比值乘以安装容量
                (pl.col("dsr_temp_0h") * pl.col("installed_capacity")).alias("ie_dsr_temp_0h"),  # 当前时刻直接太阳辐射与温度比值乘以安装容量
                (pl.col("dsr_temp_local_48h") * pl.col("installed_capacity")).alias("ie_dsr_temp_local_48h"),  # 48小时前直接太阳辐射与温度比值乘以安装容量
                (pl.col("dsr_temp_48h") * pl.col("installed_capacity")).alias("ie_dsr_temp_48h"),  # 48小时前直接太阳辐射与温度比值乘以安装容量
                (pl.col("dsr_temp_local_168h") * pl.col("installed_capacity")).alias("ie_dsr_temp_local_168h"),  # 168小时前直接太阳辐射与温度比值乘以安装容量
                (pl.col("dsr_temp_168h") * pl.col("installed_capacity")).alias("ie_dsr_temp_168h"),  # 168小时前直接太阳辐射与温度比值乘以安装容量
            )
            .with_columns(
                (pl.col("ie_dsr_temp_local_0h") - pl.col("ie_dsr_temp_local_48h")).alias("diff_ie_dsr_temp_local_0h_48h"),  # 当前时刻与48小时前直接太阳辐射与温度比值乘以安装容量的差值
                (pl.col("ie_dsr_temp_local_0h") - pl.col("ie_dsr_temp_local_168h")).alias("diff_ie_dsr_temp_local_0h_168h"),  # 当前时刻与168小时前直接太阳辐射与温度比值乘以安装容量的差值
                (pl.col("ie_dsr_temp_0h") - pl.col("ie_dsr_temp_48h")).alias("diff_ie_dsr_temp_0h_48h"),  # 当前时刻与48小时前直接太阳辐射与温度比值乘以安装容量的差值
                (pl.col("ie_dsr_temp_0h") - pl.col("ie_dsr_temp_168h")).alias("diff_ie_dsr_temp_0h_168h"),  # 当前时刻与168小时前直接太阳辐射与温度比值乘以安装容量的差值
            )
        )

        # 将polars DataFrame转为pandas DataFrame
        df_features = self._to_pandas(df_features, y)

        return df_features



class Model:
    def __init__(self):
        self.name = "ModelMixLGBMRegresssorTargetAndTargetDiff"
        self.n_models = 1
        # 模型是否已训练
        self.is_fitted = False

        # 各种消费模型的参数
        self.model_parameters_consumption = {
            "n_estimators": 1800,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_consumption_diff_168 = {
            "n_estimators": 2400,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_consumption_diff_48 = {
            "n_estimators": 3000,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_consumption_diff_mean_2 = {
            "n_estimators": 1800,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_consumption_diff_mean = {
            "n_estimators": 2400,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        # 各种生产模型的参数
        self.model_parameters_production_diff_48 = {
            "n_estimators": 3000,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_production_diff_mean_2 = {
            "n_estimators": 3000,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_production_diff_mean = {
            "n_estimators": 3000,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        self.model_parameters_production_norm = {
            "n_estimators": 400,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": "gpu",
        }
        
        # >> 消费模型
        # 用于消费预测的基本模型
        self.model_consumption = VotingRegressor(
            [
                (
                    f"consumption_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_consumption, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于消费预测，考虑168小时滞后差异的模型
        self.model_consumption_diff_168 = VotingRegressor(
            [
                (
                    f"consumption_diff_168_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_consumption_diff_168, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于消费预测，考虑48小时滞后差异的模型
        self.model_consumption_diff_48 = VotingRegressor(
            [
                (
                    f"consumption_diff_48_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_consumption_diff_48, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于消费预测，考虑滞后2-7天target的均值差异的模型
        self.model_consumption_diff_mean_2 = VotingRegressor(
            [
                (
                    f"consumption_mean_2_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_consumption_diff_mean_2, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于消费预测，考虑滞后2-5天target的均值差异的模型
        self.model_consumption_diff_mean = VotingRegressor(
            [
                (
                    f"consumption_mean_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_consumption_diff_mean, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # >> 生产模型
        # 用于生产预测，考虑48小时滞后差异的模型
        self.model_production_diff_48 = VotingRegressor(
            [
                (
                    f"production_diff_48_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_production_diff_48, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于生产预测，考虑滞后2-7天target的均值差异的模型
        self.model_production_diff_mean_2 = VotingRegressor(
            [
                (
                    f"production_diff_mean_2_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_production_diff_mean_2, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于生产预测，考虑滞后2-5天target的均值差异的模型
        self.model_production_diff_mean = VotingRegressor(
            [
                (
                    f"production_diff_mean_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_production_diff_mean, random_state=i),
                )
                for i in range(self.n_models)
            ]
        )

        # 用于生产预测，target / installed_capacity 标准化后的模型
        self.model_production_norm = VotingRegressor(
            [
                (
                    f"production_norm_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters_production_norm, random_state=i),
                )
                for i in range(5)
            ]
        )

    def fit(self, df_train_features):
        # 根据 is_consumption 将数据分为消费数据和生产数据
        mask = df_train_features["is_consumption"] == 1
        # 对消费数据进行模型训练
        self.model_consumption.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"], # 消费数据的目标值
        )
        self.model_consumption_diff_168.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_168h"].fillna(0), # 168小时滞后差异
        )
        self.model_consumption_diff_48.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_48h"].fillna(0), # 48小时滞后差异
        )
        self.model_consumption_diff_mean_2.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_mean_2"].fillna(0), # 滞后2-7天target的均值差异
        )
        self.model_consumption_diff_mean.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_mean"].fillna(0), # 滞后2-5天target的均值差异
        )

        # 对生产数据进行模型训练  
        mask = df_train_features["is_consumption"] == 0
        self.model_production_diff_48.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_48h"].fillna(0), # 48小时滞后差异
        )
        self.model_production_diff_mean_2.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_mean_2"].fillna(0), # 滞后2-7天target的均值差异
        )
        self.model_production_diff_mean.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] - df_train_features[mask]["target_mean"].fillna(0), # 滞后2-5天target的均值差异
        )
        self.model_production_norm.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"] / df_train_features[mask]["installed_capacity"].fillna(1000), # target / installed_capacity 标准化后的目标值
        )

        self.is_fitted = True

    def predict(self, df_features):
        predictions = np.zeros(len(df_features))

        # 对消费数据进行预测
        mask = df_features["is_consumption"] == 1  
        predictions[mask.values] = np.clip(
            self.model_consumption.predict(df_features[mask]) * 0.3
            + (
                df_features[mask]["target_168h"].fillna(0).values
                + self.model_consumption_diff_168.predict(df_features[mask])
            ) * 0.25
            + (
                df_features[mask]["target_48h"].fillna(0).values
                + self.model_consumption_diff_48.predict(df_features[mask])
            ) * 0.1
            + (
                df_features[mask]["target_mean_2"].fillna(0).values
                + self.model_consumption_diff_mean_2.predict(df_features[mask])
            ) * 0.25
            + (
                df_features[mask]["target_mean"].fillna(0).values
                + self.model_consumption_diff_mean.predict(df_features[mask])
            ) * 0.1
            ,
            0,
            np.inf,
        )

        # 对生产数据进行预测
        mask = df_features["is_consumption"] == 0
        predictions[mask.values] = np.clip(
            (
                df_features[mask]["target_48h"].fillna(0).values
                + self.model_production_diff_48.predict(df_features[mask])
            ) * 0.15
            + (
                df_features[mask]["installed_capacity"].fillna(1000).values
                * self.model_production_norm.predict(df_features[mask]) 
            ) * 0.5
            + (
                df_features[mask]["target_mean_2"].fillna(0).values
                + self.model_production_diff_mean_2.predict(df_features[mask])
            ) * 0.25
            + (
                df_features[mask]["target_mean"].fillna(0).values
                + self.model_production_diff_mean.predict(df_features[mask])
            ) * 0.1
            ,
            0,
            np.inf,
        )

        return predictions

# 初始化数据存储对象和特征生成器
data_storage = DataStorage()  
features_generator = FeaturesGenerator(data_storage=data_storage)

model = Model()

# 使用提供的时间序列API进行迭代预测
env = enefit.make_env()
iter_test = env.iter_test()

def is_prediciton_needed(df_test):
    return not all(df_test['currently_scored'] == False)

is_prediction_period_started = False

for (
    df_test, 
    df_new_target, 
    df_new_client, 
    df_new_historical_weather,
    df_new_forecast_weather, 
    df_new_electricity_prices, 
    df_new_gas_prices, 
    df_sample_prediction
) in iter_test:

    # 使用新数据更新数据存储对象
    data_storage.update_with_new_data(
        df_new_client=df_new_client,
        df_new_gas_prices=df_new_gas_prices,
        df_new_electricity_prices=df_new_electricity_prices,
        df_new_forecast_weather=df_new_forecast_weather,
        df_new_historical_weather=df_new_historical_weather,
        df_new_target=df_new_target
    )
    
    # 如果当前不需要预测,则直接提交一个全0的样本预测  
    if not is_prediciton_needed(df_test):
        df_sample_prediction['target'] = 0
        env.predict(df_sample_prediction)
        continue
    
    # 如果模型还没有训练,则使用历史目标数据训练模型
    if not model.is_fitted:
        df_train_features = features_generator.generate_features(data_storage.df_target)
        df_train_features = df_train_features[df_train_features['target'].notnull()] 
        model.fit(df_train_features)
        
    # 预处理测试数据    
    df_test = data_storage.preprocess_test(df_test)
    
    # 为测试数据生成特征
    df_test_features = features_generator.generate_features(df_test)
    
    # 使用训练好的模型进行预测
    df_sample_prediction["target"] = model.predict(df_test_features)
    
    # 提交预测结果
    env.predict(df_sample_prediction)

