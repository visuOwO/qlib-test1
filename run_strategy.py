import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
import os

# ================= 配置区域 =================
# 1. 指向您刚才生成的 bin 文件夹 (请确保路径正确！)
#    如果您是在当前目录下运行，通常就是 "./qlib_bin_data"
PROVIDER_URI = "./qlib_bin_data"

# 2. 设置回测和训练时间
#    注意：必须在您下载的数据时间范围内
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-11-01" # 留一点余量，不要顶格到最后一天

# ===========================================

def run_workflow():
    # 1. 初始化 Qlib
    # mount_path 是挂载路径，这样 Qlib 才知道去哪里读 bin 文件
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
    print(f"Qlib initialized with data from: {PROVIDER_URI}")

    # 2. 核心配置字典
    config = {
        # --- A. 数据与模型配置 ---
        "task": {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                },
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": TRAIN_START,
                            "end_time": TEST_END,
                            "fit_start_time": TRAIN_START,
                            "fit_end_time": TRAIN_END,
                            # 关键点：使用 "all" 代表所有本地数据
                            "instruments": "all",
                            
                            # === 核心修改：添加数据预处理 (Standardization) ===
                            # infer_processors: 用于处理因子数据 (X)
                            "infer_processors": [
                                # 1. 鲁棒的 Z-Score 标准化 (去极值 + 标准化)
                                # fit_start_time 和 fit_end_time 决定了用哪段时间的数据来计算统计量
                                # 但对于 CSZScore (截面)，它只看当天，所以时间段不敏感
                                {"class": "RobustZScore", "kwargs": {"fit_start_time": TRAIN_START, "fit_end_time": TRAIN_END, "clip_outlier": True}},

                                # 2. 缺失值填充 (这一点也很重要，否则 LightGBM 可能会处理不好 NaN)
                                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                            ],

                            # learn_processors: 用于处理标签数据 (Y, 即收益率)
                            # 预测目标通常也需要截面标准化，这能消除大盘涨跌的影响，让模型专注于选股(Alpha)
                            "learn_processors": [
                                # drop_label: 如果某天收益率缺失，就不训练这条数据
                                {"class": "DropnaLabel"},
                    
                                # CSRankNorm: 把收益率转换成排名 (0~1之间)
                                # 这样模型预测的就是“排名”而不是“具体涨了多少点”
                                # 在 A 股这种波动巨大的市场，预测排名的效果远好于预测绝对收益
                                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                            ],
                            # ===============================================
                        },
                    },
                    "segments": {
                        "train": (TRAIN_START, TRAIN_END),
                        "valid": (TEST_START, TEST_END), # 这里偷懒把验证集和测试集设一样
                        "test":  (TEST_START, TEST_END),
                    },
                },
            },
        },

        # --- B. 回测配置 ---
        "port_analysis_config": {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "signal": "<PRED>",
                    "topk": 5,   # Demo数据少，我们每天只持仓前5只
                    "n_drop": 2, # 每天轮换2只
                    "only_tradable": True,
                },
            },
            "backtest": {
                "start_time": TEST_START,
                "end_time": TEST_END,
                "account": 1000000, # 100万初始资金
                "benchmark": 'sz000066',  # Demo数据少，先不设置Benchmark，否则容易报错
                "exchange_kwargs": {
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        },
    }

    # 3. 开始运行
    print("Step 1: Training Model (LightGBM)...")
    # 初始化模型
    model = init_instance_by_config(config["task"]["model"])
    # 初始化数据集
    dataset = init_instance_by_config(config["task"]["dataset"])
    
    # 训练
    model.fit(dataset)
    print("Training finished.")

    # 预测
    print("Step 2: Predicting...")
    pred_score = model.predict(dataset)
    # 将预测结果保存为 pickle，方便后续查看
    pred_path = "pred_score.pkl"
    pred_score.to_pickle(pred_path)
    print(f"Prediction saved to {pred_score.shape}, file: {pred_path}")

    # 4. 生成信号记录 (Backtest 需要)
    print("Step 3: Generating Signals...")
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # 5. 执行回测
    print("Step 4: Running Backtest...")
    par = PortAnaRecord(recorder, config["port_analysis_config"], risk_analysis_freq="day")
    par.generate()

    # 获取回测结果保存路径
    # === 替换开始 ===
    try:
        # 尝试获取实验 ID 和 运行 ID
        exp_id = recorder.info.get('experiment_id', '0')
        run_id = recorder.info.get('id', '0')
        
        # 拼接默认路径: D:\Finance\qlib_test1\mlruns\实验ID\运行ID
        result_dir = os.path.join(os.getcwd(), "mlruns", str(exp_id), str(run_id))
        
        # 再次确认 artifacts 文件夹
        if os.path.exists(os.path.join(result_dir, "artifacts")):
            result_dir = os.path.join(result_dir, "artifacts", "portfolio_analysis")
            
    except Exception as e:
        # 如果实在找不到，就指向默认目录
        result_dir = "./mlruns"
        print(f"Warning: Could not determine exact path ({e})")

    print(f"\n==================================================")
    print(f"Workflow COMPLETED!")
    print(f"Results are saved in: {result_dir}")
    print(f"Look for 'report_normal_1day.pkl' in that folder.")
    print(f"==================================================")

    return result_dir

if __name__ == "__main__":
    run_workflow()