import pandas as pd
import numpy as np
import os
import time
from qipy import q
import multiprocessing
import gc
import schedule
from datetime import datetime

# ---------------------- 原始代码部分 --------------------------
# 参数设置
# 如果每天只更新当天数据，这里的 start_date 可只传当天，或者单独处理
# 这里假设每天更新当天数据，或者可以做历史补全
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

def calculate_synergy_matrix(data, codes):
    """利用矢量化运算计算协同矩阵"""
    # 计算各个符号变化
    return_signs = data.pivot(index='date_time', columns='code', values='minute_return_sign').values
    vol_signs = data.pivot(index='date_time', columns='code', values='vol_change_sign').values

    # 计算符号变化协同
    return_match = (return_signs[:, :, None] == return_signs[:, None, :])
    vol_match = (vol_signs[:, :, None] == vol_signs[:, None, :])

    del return_signs, vol_signs
    gc.collect()

    # 计算其他符号变化的协同次数
    return_sign_5min_change = data.pivot(index='date_time', columns='code', values='return_sign_5min_change').values
    vol_sign_5min_change = data.pivot(index='date_time', columns='code', values='vol_sign_5min_change').values

    return_sign_5min_change_match = (return_sign_5min_change[:, :, None] == return_sign_5min_change[:, None, :])
    vol_sign_5min_change_match = (vol_sign_5min_change[:, :, None] == vol_sign_5min_change[:, None, :])
    del return_sign_5min_change, vol_sign_5min_change
    gc.collect()

    # 协同矩阵的计算公式（可根据实际需求调整）
    synergy_matrix = (vol_match.sum(axis=0) + 
                      return_sign_5min_change_match.sum(axis=0) + 
                      vol_sign_5min_change_match.sum(axis=0)) + (
                      (vol_match | return_match).sum(axis=0) +
                      (return_sign_5min_change_match | vol_sign_5min_change_match).sum(axis=0))
    return synergy_matrix

def calculate_daily_synergy_diff(data, synergy_matrix, codes, trading_date, top_n=30):
    """基于协同矩阵计算日协同价差"""
    daily_results = []
    for code_idx, code in enumerate(codes):
        # 找到协同次数最多的 top_n 股票（不包括自身）
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
        data['minute_return_sign'] = np.sign(data['minute_return']).fillna(0)
        data['vol_change_sign'] = np.sign(data['minute_vol_change']).fillna(0)
        
        # 计算当前1分钟收益率相对前5分钟均值的变化符号
        data['minute_return_5min_mean'] = data.groupby('code')['minute_return'].shift(1).rolling(5).mean()
        data['return_sign_5min_change'] = np.sign(data['minute_return'] - data['minute_return_5min_mean'])
        
        # 对成交量计算类似的指标
        data['minute_vol_5min_mean'] = data.groupby('code')['vol'].shift(1).rolling(5).mean()
        data['vol_sign_5min_change'] = np.sign(data['vol'] - data['minute_return_5min_mean'])
        del data['minute_return_5min_mean'], data['minute_vol_5min_mean']
        gc.collect()
        
        # 获取所有股票代码
        codes = data['code'].unique()

        # 计算协同矩阵
        synergy_matrix = calculate_synergy_matrix(data, codes)
        del data
        gc.collect()

        # 获取日线数据，包括 'pct_change'
        d1_data = q.get_price('*', 'date,code,pct_change', trading_date, trading_date)['Data']
        d1_data = pd.DataFrame(d1_data)

        # 计算协同效应超额收益率
        daily_excess = calculate_daily_synergy_diff(d1_data, synergy_matrix, codes, trading_date)
        return daily_excess
    except Exception as e:
        print(f"Error processing data for {trading_date}: {e}")
        return pd.DataFrame()

def calculate_monthly_factors(data):
    """计算“月均成交量协同”和“月稳成交量协同”"""
    try:
        # 确保按 'date' 排序，以便进行滚动计算
        data = data.sort_values(by=['code', 'date'])
        monthly_data = data.groupby('code').rolling(20, on='date', min_periods=20).agg(
            {'daily_excess': ['mean', 'std']}).reset_index()
        monthly_data.columns = ['code', 'date', 'monthly_mean', 'monthly_std']
        monthly_data['monthly_mean'] = monthly_data['monthly_mean'].where(monthly_data.groupby('code')['monthly_mean'].transform('count') >= 20, np.nan)
        monthly_data['monthly_std'] = monthly_data['monthly_std'].where(monthly_data.groupby('code')['monthly_std'].transform('count') >= 20, np.nan)
        monthly_data['volume_collaboration'] = (monthly_data['monthly_mean'] + monthly_data['monthly_std']) / 2
        return monthly_data
    except Exception as e:
        print(f"Error calculating monthly factors: {e}")
        return pd.DataFrame()

# ---------------------- 调度更新逻辑 --------------------------

def update_daily_factor():
    """
    每天盘后运行，更新当天的因子数据，并保存日数据及月因子数据。
    根据需求可以只更新当天数据，然后将新的日数据追加到历史数据中，
    或者重新计算月因子，这里示例为当天数据和对应月因子的计算。
    """
    trading_date = datetime.today().strftime("%Y-%m-%d")
    print(f"=== 开始更新 {trading_date} 的因子数据 ===")
    
    # 处理当天数据
    daily_data = process_single_day(trading_date)
    if daily_data.empty:
        print(f"{trading_date} 无有效数据，更新结束。")
        return

    # 保存当天日因子数据（可按日期保存到单独文件，也可追加到一个总文件中）
    daily_output_file = os.path.join(output_folder, f"daily_synergy_diff_{trading_date}.csv")
    daily_data.to_csv(daily_output_file, index=False)
    print(f"当天因子数据已保存到：{daily_output_file}")

    # 若需要计算月因子，则需要有一段历史日数据。这里假设每天更新时，将当天数据与之前保存的日数据合并后计算月因子
    history_file = os.path.join(output_folder, "new_daily_synergy_diff_factors_2.csv")
    if os.path.exists(history_file):
        historical_daily = pd.read_csv(history_file)
        # 追加当天数据（注意日期格式需统一）
        combined_daily = pd.concat([historical_daily, daily_data], ignore_index=True)
    else:
        combined_daily = daily_data

    # 保存合并后的日数据（或者保留历史数据在数据库中）
    combined_daily.to_csv(history_file, index=False)
    print(f"合并后的日数据保存到：{history_file}")

    # 计算月因子（这里的计算基于所有历史日数据，实际可根据需求限定日期范围）
    monthly_data = calculate_monthly_factors(combined_daily)
    monthly_output_file = os.path.join(output_folder, f"new_monthly_synergy_diff_factors_{trading_date}.csv")
    monthly_data.to_csv(monthly_output_file, index=False)
    print(f"月因子数据已保存到：{monthly_output_file}")

    print(f"=== {trading_date} 因子数据更新完成 ===")

def job():
    """调度任务的入口函数"""
    try:
        update_daily_factor()
    except Exception as e:
        print(f"更新任务出现异常：{e}")

if __name__ == '__main__':
    # 设置每天盘后（例如16:00）执行任务
    schedule_time = "16:00"  # 根据实际盘后时间调整
    schedule.every().day.at(schedule_time).do(job)
    print(f"调度任务已设置，每天 {schedule_time} 运行因子更新。")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每60秒检测一次
    except KeyboardInterrupt:
        print("程序手动终止。")
    finally:
        q.stop()  # 停止数据接口连接
