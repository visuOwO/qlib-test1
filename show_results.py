import pandas as pd
import matplotlib.pyplot as plt

# 【重要】请把这里替换为您刚才 run_strategy.py 最后输出的那个路径
# 例如: report_path = "mlruns/1/e5a8.../artifacts/portfolio_analysis/report_normal_1day.pkl"
report_path = "mlruns/525301740681149810/327fd2945e9644d39e3df2e79568cf71/artifacts/portfolio_analysis/report_normal_1day.pkl" 

try:
    # 读取回测报告
    df = pd.read_pickle(report_path)
    
    # 打印前几行看看
    print("回测数据预览:")
    print(df.head())
    
    # 计算累计收益
    # return 是日收益率，cumulative return 是累乘
    df['cumulative_return'] = (1 + df['return']).cumprod()
    
    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_return'], label='My Strategy')
    plt.title('Backtest Cumulative Return')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"年化收益率: {df['return'].mean() * 252 * 100:.2f}%")

except FileNotFoundError:
    print("找不到文件，请检查 report_path 路径是否正确。")