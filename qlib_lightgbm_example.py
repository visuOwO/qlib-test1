import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.tests.data import GetData

# Initialize qlib with a temporary local file provider
provider_uri = "~/.qlib/qlib_data/cn_data"
GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
qlib.init(provider_uri=provider_uri, region=REG_CN)

# Configuration for the model and backtesting
config = {
    "task": {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.lightgbm",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "n_estimators": 200,
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
                        "start_time": "2008-01-01",
                        "end_time": "2020-08-01",
                        "fit_start_time": "2008-01-01",
                        "fit_end_time": "2014-12-31",
                        "instruments": "csi300",
                    },
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    },
    "backtest": {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "topk": 50,
                "n_drop": 5,
            },
        },
        "executor": {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "exchange": {
            "exchange_bin": None,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# Run the workflow
with R.start(experiment_name="lightgbm_example"):
    # Initialize components from the configuration
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    # Train the model
    R.log_params(**config["task"]["model"]["kwargs"])
    model.fit(dataset)
    R.save_objects(trained_model=model)

    # Prediction
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # Backtesting
    par = PortAnaRecord(recorder, config["backtest"], "day")
    par.generate()

    print("Workflow finished. Results are saved in the experiment.")
