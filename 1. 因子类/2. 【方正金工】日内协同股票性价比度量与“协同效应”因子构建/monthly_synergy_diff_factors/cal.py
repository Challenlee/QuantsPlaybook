import pandas as pd
import numpy as np
import os
import time
from qipy import q
import multiprocessing
import gc

# 参数设置
start_date = '2014-01-01'
end_date = '2025-01-12'
output_folder = 'final_results'
os.makedirs(output_folder, exist_ok=True)

# 启动数据接口
q.start()

def fetch_daily_data(trading_date):
    """获取指定日期的所有股票分钟数据"""
    try:
        print(f"Fetching data for {trading_date}...")
        data = q.get_price('*', 'code,close,vol,date_time',
                           trading_date + ' 09:00:00', trading_date + ' 15:30:30', "freq=m1")['Data']
        data = pd.DataFrame(data)
        data['date_time'] = pd.to_datetime(data['date_time'])
        return data
    except Exception as e:
        print(f"Error fetching data for {trading_date}: {e}")
        return pd.DataFrame()
#-----------------------------------------------------------------------------------------

def calculate_synergy_matrix(data, codes):
    """利用矢量化运算计算协同矩阵"""
    # 计算各个符号变化
    return_signs = data.pivot(index='date_time', columns='code', values='minute_return_sign').values
    vol_signs = data.pivot(index='date_time', columns='code', values='vol_change_sign').values

    # 计算符号变化协同
    return_match = (return_signs[:, :, None] == return_signs[:, None, :])
    vol_match = (vol_signs[:, :, None] == vol_signs[:, None, :])

    del return_signs
    del vol_signs
    gc.collect()
    # 计算其他符号变化的协同次数
    # return_sign_change_match = data.pivot(index='date_time', columns='code', values='return_sign_change').values
    return_sign_5min_change = data.pivot(index='date_time', columns='code', values='return_sign_5min_change').values
    vol_sign_5min_change = data.pivot(index='date_time', columns='code', values='vol_sign_5min_change').values

    return_sign_5min_change_match = (return_sign_5min_change[:, :, None] == return_sign_5min_change[:, None, :])
    vol_sign_5min_change_match = (vol_sign_5min_change[:, :, None] == vol_sign_5min_change[:, None, :])
    del return_sign_5min_change
    del vol_sign_5min_change
    gc.collect()
    # print(vol_sign_5min_change_match)
    # synergy_matrix += (return_sign_change_match | return_sign_5min_change_match | vol_sign_5min_change_match).sum(axis=0)
    # 计算符号变化的协同次数
    synergy_matrix = ( (vol_match).sum(axis=0)+ (return_sign_5min_change_match).sum(axis=0) + (vol_sign_5min_change_match).sum(axis=0)) +(
        vol_match|return_match).sum(axis=0)+( return_sign_5min_change_match|vol_sign_5min_change_match).sum(axis=0)
    # print(synergy_matrix)
    return synergy_matrix
#-----------------------------------------------------------------------------------------

def calculate_daily_synergy_diff(data, synergy_matrix, codes, trading_date, top_n=30):
    """基于协同矩阵计算日协同价差"""
    daily_results = []
    for code_idx, code in enumerate(codes):
        # 找到协同次数最多的 top_n 股票
        synergy_counts = synergy_matrix[code_idx]
        top_synergistic_codes = codes[np.argsort(-synergy_counts)[1:top_n+1]]

        # 当前股票的收益率
        stock_return = data.loc[data['code'] == code, 'pct_change'].mean()

        # 协同股票的平均收益率
        top_avg_return = data[data['code'].isin(top_synergistic_codes)]['pct_change'].mean()

        # 计算超额收益率
        daily_excess = stock_return - top_avg_return

        # 保存结果，包括日期
        daily_results.append({'date': trading_date, 'code': code, 'daily_excess': daily_excess})
    return pd.DataFrame(daily_results)
#-----------------------------------------------------------------------------------------

def process_single_day(trading_date):
    """处理单日数据"""
    try:
        print(f"Processing data for {trading_date}...")

        # 获取分钟数据
        data = fetch_daily_data(trading_date)
        if data.empty:
            print(f"No data found for {trading_date}. Skipping...")
            return pd.DataFrame()

        # 计算收益率与符号
        data = data.sort_values(by=['code', 'date_time'])
        data['minute_return'] = data.groupby('code')['close'].pct_change()
        data['minute_vol_change'] = data.groupby('code')['vol'].pct_change()
        data['minute_return_sign'] = np.sign(data['minute_return']).fillna(0) ## 最近一分钟价格、vol变化方向是否一致
        data['vol_change_sign'] = np.sign(data['minute_vol_change']).fillna(0)
        
        # 这里的计算错了，是计算当前1分钟收益率相对前5分钟收益率（不包括当前分钟）的变化符号
        # 计算当前1分钟收益率相对前5分钟收益率均值的变化符号
        data['minute_return_5min_mean'] = data.groupby('code')['minute_return'].shift(1).rolling(5).mean()
        data['return_sign_5min_change'] = np.sign(data['minute_return'] - data['minute_return_5min_mean'])
        
        # data['vol_sign_5min'] = data.groupby('code')['vol_change_sign'].apply(
        #     lambda x: x.shift(1).rolling(5).apply(lambda y: np.sign(y).mean(), raw=True)
        # )
        # data['vol_sign_5min_change'] = data['vol_change_sign'] == data['vol_sign_5min']
        data['minute_vol_5min_mean'] = data.groupby('code')['vol'].shift(1).rolling(5).mean()
        data['vol_sign_5min_change'] = np.sign(data['vol'] - data['minute_return_5min_mean'])
        del data['minute_return_5min_mean']
        del data['minute_vol_5min_mean']
        gc.collect()
        # 获取所有股票代码
        codes = data['code'].unique()

        # 计算协同矩阵
        synergy_matrix = calculate_synergy_matrix(data, codes)
        del data
        gc.collect()

        # 获取 1 日数据，包括 'pct_change'
        d1_data = q.get_price('*', 'date,code,pct_change', trading_date, trading_date)['Data']
        d1_data = pd.DataFrame(d1_data)

        # 计算协同效应超额收益率
        daily_excess = calculate_daily_synergy_diff(d1_data, synergy_matrix, codes, trading_date)

        return daily_excess
    except Exception as e:
        print(f"Error processing data for {trading_date}: {e}")
        return pd.DataFrame()
#-----------------------------------------------------------------------------------------
def calculate_monthly_factors(data):
    """计算“月均成交量协同”和“月稳成交量协同”"""
    try:
        # 确保按 'date' 排序，以便进行滚动计算
        data = data.sort_values(by=['code', 'date'])

        # 对每个 'code' 计算过去20天的均值和标准差
        monthly_data = data.groupby('code').rolling(20, on='date', min_periods=20).agg(
            {'daily_excess': ['mean', 'std']}).reset_index()

        # 清理多级列，并为列命名
        monthly_data.columns = ['code', 'date', 'monthly_mean', 'monthly_std']

        # 如果滚动窗口不足20天，设为 NaN
        monthly_data['monthly_mean'] = monthly_data['monthly_mean'].where(monthly_data.groupby('code')['monthly_mean'].transform('count') >= 20, np.nan)
        monthly_data['monthly_std'] = monthly_data['monthly_std'].where(monthly_data.groupby('code')['monthly_std'].transform('count') >= 20, np.nan)

        # 计算“成交量协同”因子 = 月均成交量协同 + 月稳成交量协同
        monthly_data['volume_collaboration'] = (monthly_data['monthly_mean'] + monthly_data['monthly_std']) / 2

        return monthly_data
    except Exception as e:
        print(f"Error calculating monthly factors: {e}")
        return pd.DataFrame()
#-----------------------------------------------------------------------------------------

def process_all_days(trading_dates):
    """处理所有交易日数据"""
    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(process_single_day, trading_dates)

    return pd.concat([res for res in results if not res.empty], ignore_index=True)
#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    overall_start_time = time.time()

    print(f"Start processing data from {start_date} to {end_date}.")
    trading_dates = q.get_day_between(start_date, end_date)

    # 处理交易日数据
    daily_data = process_all_days(trading_dates)

    if not daily_data.empty:
        # 计算月均协同价差和月稳协同价差
        monthly_data = calculate_monthly_factors(daily_data)

        # 保存结果
        daily_output_file = os.path.join(output_folder, "new_daily_synergy_diff_factors_2.csv")
        monthly_output_file = os.path.join(output_folder, "new_monthly_synergy_diff_factors_2.csv")

        daily_data.to_csv(daily_output_file, index=False)
        monthly_data.to_csv(monthly_output_file, index=False)
        print(f"Results saved to {daily_output_file, monthly_output_file}.")
    else:
        print("No data processed.")

    print(f"Total time taken: {time.time() - overall_start_time:.2f} seconds.")
    q.stop()