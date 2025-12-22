import tushare as ts
import pandas as pd
import os
import time
from pathlib import Path
from token_manager import get_valid_token
import numpy as np

# ================= 配置区域 =================
# 1. 请替换为您的 Tushare Token
TS_TOKEN = get_valid_token()

# 2. 设置数据范围 (建议先用短时间、少量股票测试跑通)
START_DATE = "20200101"
END_DATE = "20251031"

# 3. 输出目录设置
CSV_OUTPUT_DIR = Path("./qlib_source_csv") # 存放转换好的CSV
QLIB_DATA_DIR = Path("./qlib_bin_data")    # 存放最终的bin文件

# ===========================================

def init_tushare():
    ts.set_token(TS_TOKEN)
    return ts.pro_api()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_stock_list(pro):
    """获取中证500成分股作为Demo (避免下载全市场耗时过长)"""
    # 如果您想下载全市场，可以使用 pro.stock_basic(exchange='', list_status='L')
    print("正在获取股票列表 (以中证500为例)...")
    df = pro.index_weight(index_code='000905.SH', start_date="20241201", end_date="20241231")
    stock_list = df['con_code'].unique().tolist()
    print(f"获取到 {len(stock_list)} 只股票。")
    return stock_list

# 预先加载行业映射
def get_industry_map(pro):
    # 获取基础信息，包含行业 (industry) 或申万行业 (sw_l1)
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
    
    # 将字符串行业转换为 ID (0, 1, 2...)
    df['industry_id'] = pd.factorize(df['industry'])[0]
    return df.set_index('ts_code')['industry_id'].to_dict()

def process_single_stock(pro, ts_code, start_date, end_date, industry_map):
    """
    下载单个股票数据并格式化
    Qlib 要求的 CSV 格式: [date, open, high, low, close, volume, amount, factor]
    """
    try:
        # 1. 获取日线行情 (Open, High, Low, Close, Vol, Amount)
        df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        # 写入行业 ID
        # 注意：行业 ID 是静态的，每天都一样，存为 float
        ind_id = industry_map.get(ts_code, np.nan)
        print(ind_id)
        
        # 2. 获取复权因子 (Adj Factor) - Qlib 需要这个来计算后复权价
        df_factor = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        # 3. 获取基础指标 (含换手率) - turnover_rate 为百分比
        fields = 'trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe_ttm,pb,ps_ttm,dv_ttm,total_mv,circ_mv'
        df_basic = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        
        if df_daily.empty or df_factor.empty:
            return None

        # 4. 合并数据
        df_merge = pd.merge(df_daily, df_factor, on='trade_date', how='inner', suffixes=('', '_y'))
        # 将换手率并入，允许缺失值
        if df_basic is not None and not df_basic.empty:
            df_merge = pd.merge(df_merge, df_basic, on='trade_date', how='left')
        
        # 5. 数据清洗与重命名
        # Tushare: trade_date, open, high, low, close, vol(手), amount(千元)
        # Qlib:    date, open, high, low, close, volume, amount, factor
        
        data = pd.DataFrame()
        data['industry'] = [ind_id] * len(df_merge)

        data['date'] = pd.to_datetime(df_merge['trade_date'])
        data['open'] = df_merge['open']
        data['high'] = df_merge['high']
        data['low'] = df_merge['low']
        data['close'] = df_merge['close']
        
        # 重要：单位转换
        # Tushare vol 是“手”，Qlib 建议转为“股” (*100)
        data['volume'] = df_merge['vol'] * 100 
        # Tushare amount 是“千元”，Qlib 建议转为“元” (*1000)
        data['amount'] = df_merge['amount'] * 1000
        
        # 复权因子
        data['factor'] = df_merge['adj_factor']
        
        # 换手率 (百分比)，可能存在缺失
        if 'turnover_rate' in df_merge.columns:
            data['turnover_rate'] = df_merge['turnover_rate']

        # 估值类
        data['pe_ttm'] = df_merge['pe_ttm']
        data['pb'] = df_merge['pb']
        data['ps_ttm'] = df_merge['ps_ttm']
        data['dv_ttm'] = df_merge['dv_ttm']

        # 情绪类
        data['turnover_rate_f'] = df_merge['turnover_rate_f']
        data['volume_ratio'] = df_merge['volume_ratio']
        
        # 市值类 (注意单位转换：Tushare 给的是万元，建议转为元)
        data['total_mv'] = df_merge['total_mv'] * 10000 
        data['circ_mv'] = df_merge['circ_mv'] * 10000
        
        # 必须按日期升序排列
        data = data.sort_values('date').reset_index(drop=True)
        
        # Qlib CSV 要求 date 格式为 YYYY-MM-DD
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        
        return data

    except Exception as e:
        print(f"Error processing {ts_code}: {e}")
        return None

def process_index_data(pro, ts_code, start_date, end_date):
    """
    下载指数数据 (如沪深300) 并格式化为 Qlib 格式
    """
    print(f"\n开始下载指数数据: {ts_code} ...")
    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            print(f"指数 {ts_code} 数据为空")
            return

        data = pd.DataFrame()
        data['date'] = pd.to_datetime(df['trade_date'])
        data['open'] = df['open']
        data['high'] = df['high']
        data['low'] = df['low']
        data['close'] = df['close']
        
        # 指数成交量通常单位也是手，成交额也是千元，保持统一转换
        data['volume'] = df['vol'] * 100
        data['amount'] = df['amount'] * 1000
        
        # 指数没有复权因子，默认为 1.0
        data['factor'] = 1.0
        
        data = data.sort_values('date').reset_index(drop=True)
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        
        # 转换文件名: 000300.SH -> sh000300.csv
        parts = ts_code.split('.')
        qlib_code = f"{parts[1].lower()}{parts[0]}"
        file_path = CSV_OUTPUT_DIR / f"{qlib_code}.csv"
        
        data.to_csv(file_path, index=False)
        print(f"指数 {ts_code} 下载完成，保存至 {file_path}")

    except Exception as e:
        print(f"Error processing index {ts_code}: {e}")

def main():
    pro = init_tushare()
    ensure_dir(CSV_OUTPUT_DIR)

    # 1. 获取行业映射
    industry_map = get_industry_map(pro)
    
    stock_list = get_stock_list(pro)
    
    # 限制 Demo 数量，防止 Tushare 触发流控 (如果您的积分够高，可以去掉 [:50])
    target_stocks = stock_list
    print(f"开始下载 {len(target_stocks)} 只股票数据...")
    
    success_count = 0
    
    for i, ts_code in enumerate(target_stocks):
        # 转换代码格式: 000001.SZ -> sz000001 (Qlib 习惯小写+前缀，虽然不做强制，但建议统一)
        # 这里为了简单，我们保持 csv 文件名为 sh600000.csv 这种风格
        parts = ts_code.split('.')
        qlib_code = f"{parts[1].lower()}{parts[0]}"
        file_path = CSV_OUTPUT_DIR / f"{qlib_code}.csv"
        
        # 检查是否已存在
        if file_path.exists():
            print(f"[{i+1}/{len(target_stocks)}] {qlib_code} 已存在，跳过。")
            success_count += 1
            continue
            
        df = process_single_stock(pro, ts_code, START_DATE, END_DATE, industry_map)
        
        if df is not None and not df.empty:
            # 保存为 CSV
            df.to_csv(file_path, index=False)
            print(f"[{i+1}/{len(target_stocks)}] {qlib_code} 下载完成")
            success_count += 1
        else:
            print(f"[{i+1}/{len(target_stocks)}] {qlib_code} 数据为空或下载失败")
        
        # Tushare 免费接口有频率限制，务必加 sleep
        time.sleep(0.3) 

    # 下载沪深300指数作为基准
    process_index_data(pro, '000300.SH', START_DATE, END_DATE)

    print(f"\nCSV 下载完成! 成功: {success_count}。文件保存在: {CSV_OUTPUT_DIR}")
    print("下一步：请运行 Qlib 的 dump_bin 命令将 CSV 转换为 Bin 文件。")

if __name__ == "__main__":
    main()