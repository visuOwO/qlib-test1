# Qlib 量化交易工作流

本项目提供了一套基于微软 [Qlib](https://github.com/microsoft/qlib) 的量化交易策略研究完整工作流。它涵盖了从数据获取（通过 Tushare）、二进制转换、模型训练（LightGBM）、回测到结果可视化的全过程。

## 前置要求

请确保您安装了以下 Python 库：

```bash
pip install pyqlib pandas tushare fire tqdm lightgbm matplotlib scikit-learn
```

*注意：您还需要一个有效的 [Tushare](https://tushare.pro/) API Token。*

## 文件结构

*   **`download_to_qlib.py`**: 从 Tushare 获取历史股票数据（日线价格、复权因子、基础指标）并将它们保存为 Qlib 标准的 CSV 文件。
*   **`dump_bin.py`**: 一个命令行工具，用于将下载的 CSV 文件转换为 Qlib 的高性能二进制格式。
*   **`run_strategy.py`**: 主入口程序。它定义了机器学习工作流：
    *   **数据处理器**: `Alpha158` (Qlib 标准因子集)。
    *   **模型**: `LGBModel` (LightGBM)。
    *   **策略**: `TopkDropoutStrategy` (Top-K 轮换策略)。
    *   **回测**: 执行交易模拟。
*   **`show_results.py`**: 可视化回测结果（累计收益、超额收益等）。
*   **`qlib_lightgbm_example.py`**: 一个独立的参考脚本，展示了 Qlib 的内部结构。
*   **`token_manager.py`**: 用于管理 Tushare API Token 的辅助工具。

## 快速开始指南

按照以下步骤运行您的第一次回测。

### 第一步：数据获取

1.  打开 `download_to_qlib.py`。
2.  确保您的 Tushare Token 有效（通过 `token_manager.py` 管理或直接硬编码）。
3.  如果需要，调整 `START_DATE`（开始日期）和 `END_DATE`（结束日期）。
4.  运行脚本：

```bash
python download_to_qlib.py
```
*输出：数据将被保存到 `./qlib_source_csv` 目录。*

### 第二步：数据转换 (Dump Binary)

将 CSV 数据转换为 Qlib 的二进制格式。

```bash
python dump_bin.py dump_all --data_path ./qlib_source_csv --qlib_dir ./qlib_bin_data --date_field_name date
```

*输出：二进制文件将在 `./qlib_bin_data` 目录中生成。*

### 第三步：运行策略

训练模型并运行回测。

1.  打开 `run_strategy.py`。
2.  检查 `PROVIDER_URI` 是否指向您的二进制数据文件夹 (`./qlib_bin_data`)。
3.  调整 `TRAIN_START`、`TRAIN_END`、`TEST_START` 和 `TEST_END` 以匹配您下载的数据范围。
4.  运行脚本：

```bash
python run_strategy.py
```

*输出：脚本将训练 LightGBM 模型，预测分数并运行回测。结果保存在 `mlruns` 目录中。*

### 第四步：可视化结果

查看您的策略表现。

```bash
python show_results.py
```

*输出：一个 matplotlib 窗口，显示您的策略与基准的累计收益对比。*

## 自定义

*   **添加因子**: 创建一个继承自 `qlib.contrib.data.handler.Alpha158` 的自定义类，并在 `get_feature_config` 中添加您的公式。然后更新 `run_strategy.py` 以使用您的新类。
*   **修改策略**: 修改 `run_strategy.py` 中的 `port_analysis_config` 部分。
