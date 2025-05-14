import pandas as pd
from qipy import q
import os
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from joblib import Parallel, delayed
import polars as pl
import datetime as dt
import time
import multiprocessing

# 参数设置
window_for_vol_mean = 20  # 成交额均值计算的滚动窗口                python upd_csp_comp.py
end_date = '2025-02-13' #        datetime.today().strftime("%Y-%m-%d")


n_days = 756   # 过去756个交易日，约3年
lag_days = 20  # 不使用最近20个交易日的数据
min_samples = 20  # 历史窗口中要求至少252个样本，否则 IR 为空
# output_folder = 'final_results'
# os.makedirs(output_folder, exist_ok=True)
stock_codes = ['*']  # 全部股票
num1 = 1
num2 = 11

overall_start_time = time.time()
q.start()
start_date = q.get_near_day(end_date,-1000)
dates = q.get_day_between(start_date, end_date)
date_list = pd.to_datetime(dates)
# print(len(dates))
def fetch_daily_data(trading_date):
    """获取指定日期的所有股票分钟数据"""
    try:
        # print(f"Fetching data for {trading_date}...")
        data = q.get_price(stock_codes, 'date,code,open_adj,high_adj,low_adj,close_adj,amt', 
                                  trading_date, trading_date)['Data']
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['date'].astype(str).str[:8])
        # data['wb'] = (data['w_buy'] -data['w_sale']) / (data['w_buy'] + data['w_sale'] + 1)
        return data
    except Exception as e:
        print(f"Error fetching data for {trading_date}: {e}")
        return pd.DataFrame()


def process_all_days(trading_dates):
    """处理所有交易日数据"""
    with multiprocessing.Pool(processes=20) as pool:
        results = pool.map(fetch_daily_data, dates)
    return pd.concat([res for res in results if not res.empty], ignore_index=True)


df_price = process_all_days( date_list)
if df_price.empty:
    print("无价格数据。")
    q.stop()
    exit()
q.stop()


df_price = pl.DataFrame(df_price)


df_price = (
    df_price
    .select(['date', 'code', 'open_adj', 'high_adj', 'low_adj', 'close_adj', 'amt'])
    .filter(~pl.col('code').str.contains('HK'))
    .with_columns(
        pl.col('code').cast(pl.Categorical),
        pl.col('date').cast(pl.Date).cast(pl.Utf8),
    # .drop_nulls()
        ((pl.col('close_adj').shift(-lag_days) / pl.col('close_adj') - 1).over('code')).alias('future20_return')
    ).filter(pl.col("amt") != 0)
)
# print(df_price)
df_000985 = df_price.filter(pl.col('code') == '000985.SZ').select(['date','future20_return'])
df_price = df_price.join(df_000985, on='date', suffix="_target") \
    .with_columns((pl.col('future20_return') - pl.col('future20_return_target')/(pl.col('future20_return_target')+1)).alias('future20_return'))


def aggregate_k_lines(df_group,i):
        result = df_price.lazy().with_columns(
            k_high=pl.col("high_adj").rolling_max(window_size=i).over("code"),
            k_low=pl.col("low_adj").rolling_min(window_size=i).over("code"),
            k_open=pl.col("open_adj").shift(i-1).over("code"),
            k_close=pl.col("close_adj"),
            amount_mean=pl.col("amt").rolling_mean(window_size=window_for_vol_mean).over("code"),

        )
        # result = result.with_columns(
        #     pl.col("k_amount").rolling_mean(window_size=window_for_vol_mean).over("code").alias("amount_mean")
        # )
        return result.select(['date','code','k_high','k_low','k_open','k_close','amt','amount_mean','future20_return']).collect()

def calculate_price_indicators(df_k_line):
        df_k_line = df_k_line.with_columns(
            pl.col("k_close").shift(1).over("code").alias("pre_close")
        )

        # 第二步：使用已存在的列计算其他指标
        df_k_line = df_k_line.with_columns(
            ((pl.col("k_open") - pl.col("k_close")).abs() / pl.col("pre_close")).alias("body"),
            ((pl.col("k_high") - pl.max_horizontal("k_open", "k_close")) / pl.col("pre_close")).alias("upper_shadow"),
            ((pl.min_horizontal("k_open", "k_close") - pl.col("k_low")) / pl.col("pre_close")).alias("lower_shadow"),
            ((pl.col("amt")) / pl.col("amount_mean").over("code")).alias("amt_ratio"),
        ).drop_nulls(subset=["lower_shadow", 'upper_shadow', 'body', 'amt_ratio'])

        # 打上标记
        df_k_line = df_k_line.with_columns(
            (pl.when(pl.col("k_close") > pl.col("k_open")).then(pl.lit("阳")).otherwise(pl.lit("阴")) +
             pl.when(pl.col("body") <= 0.015).then(pl.lit("小"))
             .when((pl.col("body") > 0.015) & (pl.col("body") <= 0.03)).then(pl.lit("中"))
             .otherwise(pl.lit("大")) +
             pl.when(pl.col("upper_shadow") > 0.02).then(pl.lit("长上")).otherwise(pl.lit("短上")) +
             pl.when(pl.col("lower_shadow") > 0.02).then(pl.lit("长下")).otherwise(pl.lit("短下"))).alias("price_pattern")
        )
        df_k_line = df_k_line.with_columns(
            pattern=(pl.when(pl.col("amt_ratio") >= 1.2).then(pl.lit("放")).when(pl.col("amt_ratio") <= 0.8).then(pl.lit("缩")).otherwise(pl.lit("常")) +
                     # 前两日和今日的price_pattern
                     pl.col("price_pattern").shift(2) + "_" + pl.col("price_pattern").shift(1) + "_" + pl.col("price_pattern")).cast(pl.Categorical)
        ).drop_nulls(subset=['pattern'])

        return df_k_line.select(['date','code','pattern','future20_return'])

def calculate_ir_metrics(df_k_line, lag_days, n_days, min_samples):
        q = (
            df_k_line.lazy().group_by(['pattern', 'date']).agg(
                pl.len().alias('count'),
                pl.sum('future20_return').alias('sum'),
                (pl.col('future20_return') ** 2).sum().alias('sumsq')
            )
            .sort("date", descending=False)
            )
        date = df_k_line.select('date').unique().sort('date')
        pattern = df_k_line.select('pattern').unique().sort('pattern')
        ir_all = pattern.join(date, how="cross").lazy()

        q = ir_all.join(q, on=["pattern", "date"], how="left").fill_null(0)
        q = q.with_columns(
                pl.col('count').shift(lag_days).over('pattern'),
                pl.col('sum').shift(lag_days).over('pattern'),
                pl.col('sumsq').shift(lag_days).over('pattern')
        )#.drop_nulls(subset=['count'])
        
        q = q.with_columns(
            rolling_sum=pl.col('sum').rolling_sum(window_size=n_days, min_periods=1),
            rolling_count=pl.col('count').rolling_sum(window_size=n_days, min_periods=1),
            rolling_sumsq=pl.col('sumsq').rolling_sum(window_size=n_days, min_periods=1)
        ).sort('date', descending=False)
        
        q = q.with_columns(
            ir=pl.col('rolling_sum') / pl.col('rolling_count') /
            ((pl.col('rolling_sumsq') / pl.col('rolling_count') - pl.col('rolling_sum') / pl.col('rolling_count') ** 2) ** 0.5)
        )
        
        q = q.with_columns(
            pl.when(pl.col('rolling_count') <= min_samples).then(None).otherwise(pl.col('ir')).alias('ir')
        ).select(['date', 'pattern', 'ir']).sort(['date',], descending=False)

        return q.collect()


def calculate_csp_factor(i):
    start_time = time.time()
    ##第一步，聚合k线
    df_k_line = aggregate_k_lines(df_price,i)
    print(f"aggregate_k_lines 耗时: {time.time() - start_time:.2f}秒")
    ## 第二步，打上标签
    df_k_line = calculate_price_indicators(df_k_line)
    ## 第三步，计算pattern的聚合指标
    

    ir = calculate_ir_metrics(df_k_line, lag_days, n_days, min_samples)
    merged_df = ir.join(
        df_k_line.select(['date','pattern','code']),
        on=["date", "pattern"],
        how="inner"
    ).select( "date", "code","ir")
    merged_df = merged_df.with_columns(
        pl.col('ir').rolling_mean(20, min_periods=5).over('code').alias(f'factor_{i}') 
    ).drop_nulls(subset=[f'factor_{i}']).sort('date', descending=False)


    return merged_df.select("date", "code", f'factor_{i}')




df = pl.DataFrame(df_price[['date','code']].unique())


for i in range(num1, num2):
    start_time = time.time()
    df_factor = calculate_csp_factor(i)
    if df.is_empty():
        df = df_factor
    else:
        df = df.join(df_factor, on=["date", "code"], how="left")
    # print(df)
    print(f"Completed i={i},耗时: {time.time() - start_time:.2f}秒")
# print(df)
df = df.fill_null(0)
df = df.with_columns(
    factor = (pl.col(f'factor_1')+pl.col(f'factor_2')+pl.col(f'factor_3')+pl.col(f'factor_4')+pl.col(f'factor_5')+pl.col(f'factor_6')+pl.col(f'factor_7')+pl.col(f'factor_8')+pl.col(f'factor_9')+pl.col(f'factor_10'))/10
).select(['date','code','factor']).filter(pl.col('factor') != 0).sort('date', descending=False).filter(pl.col('date')==end_date).sort('code')
print(df)
df = df.to_pandas()

print(f"总耗时: {time.time() - overall_start_time:.2f}秒")