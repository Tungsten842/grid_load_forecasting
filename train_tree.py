from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_features(lf, group_id, ts_name, target_name):
    ts = pl.col(ts_name)
    target = pl.col(target_name)

    time_features = [
        (ts.dt.hour() * 2 + ts.dt.minute() // 30).alias("half_hour_period"),
        ts.dt.weekday().alias("day_of_week"),
        ts.dt.month().alias("month"),
    ]

    lags = {"lag_30min": 1, "lag_1h": 2, "lag_24h": 48, "lag_7d": 336}
    lag_features = [
        target.shift(steps).over(group_id).alias(name) for name, steps in lags.items()
    ]

    new_features = time_features + lag_features
    return lf.with_columns(new_features)


def prepare_dataset(file, n_series=None, return_unique_id=False):
    group_id = "lv_feeder_unique_id"
    ts_name = "data_collection_log_timestamp"
    target_name = "mean_consumption_per_meter"
    device_count_name = "aggregated_device_count_active"

    raw_columns = [group_id, ts_name, target_name, device_count_name]

    lf = pl.scan_parquet(file).select(raw_columns)

    if n_series:
        ids = lf.select(group_id).unique().head(n_series)
        lf = lf.join(ids, on=group_id)

    lf = prepare_features(lf, group_id, ts_name, target_name)

    drop_cols = [ts_name]
    if not return_unique_id:
        drop_cols.append(group_id)

    df = lf.drop(drop_cols).drop_nulls().collect()

    X = df.drop(target_name)
    y = df.get_column(target_name)

    return X, y


class MultiQuantileLGBM:
    def __init__(self, quantiles, **params):
        self.quantiles = quantiles
        self.params = params
        self.models = {}

    def fit(self, *args, **kwargs):
        train_data = lgb.Dataset(args[0], label=args[1]).construct()

        val_data = lgb.Dataset(
            kwargs.pop("eval_X"),
            label=kwargs.pop("eval_y"),
            reference=train_data,
        ).construct()

        for q in self.quantiles:
            self.models[q] = lgb.train(
                {**self.params, "objective": "quantile", "alpha": q},
                train_set=train_data,
                valid_sets=[val_data],
                **kwargs,
            )
        return self

    def predict(self, X):
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        return preds


def main():
    path = Path("dataset")
    X_train, y_train = prepare_dataset(path / "train.parquet")
    X_val, y_val = prepare_dataset(path / "val.parquet")
    X_test, y_test = prepare_dataset(path / "test.parquet")

    model = MultiQuantileLGBM(
        quantiles=(0.1, 0.5, 0.9),
        force_row_wise=True,
        data_sample_strategy="goss",
        n_estimators=15,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(1), lgb.log_evaluation(1)],
    )
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds[:, 1]))
    mae = mean_absolute_error(y_test, preds[:, 1])

    print(f"\nReal Test RMSE: {rmse:,.2f}")
    print(f"Real Test MAE:  {mae:,.2f}")

    joblib.dump(model, "lgb_model.pkl")


if __name__ == "__main__":
    main()
