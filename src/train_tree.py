from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import mean_pinball_loss


def prepare_features(lf, group_id, ts_name, target_name):
    ts = pl.col(ts_name)
    target = pl.col(target_name)

    temp = pl.col("temperature")
    hum = pl.col("humidity")
    weather_features = [
        (temp - ((100.0 - hum) / 5.0)).alias("dew_point"),
        (temp - 0.55 * (1.0 - hum / 100.0) * (temp - 14.5)).alias("feels_like_temp"),
    ]

    time_features = [
        (ts.dt.hour() * 2 + ts.dt.minute() // 30).alias("half_hour_period"),
        ts.dt.weekday().alias("day_of_week"),
        ts.dt.month().alias("month"),
    ]

    l7 = target.shift(336).over(group_id)
    l8 = target.shift(384).over(group_id)
    l14 = target.shift(672).over(group_id)

    lag_features = [
        l7.alias("lag_7d"),
        l8.alias("lag_8d"),
        l14.alias("lag_14d"),
    ]

    derived_features = [
        (l7 - l14).alias("diff_7d_14d"),
        (l7 - l8).alias("diff_7d_8d"),
    ]

    rolling_features = [
        target.shift(336)
        .rolling_mean(336)
        .over(group_id)
        .cast(pl.Float32)
        .alias("rolling_mean_7d"),
        target.shift(336)
        .rolling_std(336)
        .over(group_id)
        .cast(pl.Float32)
        .alias("rolling_std_7d"),
    ]

    new_features = (
        weather_features
        + time_features
        + lag_features
        + rolling_features
        + derived_features
    )
    return lf.with_columns(new_features)


def prepare_dataset(file, n_series=None, return_unique_id=False):
    group_id = "lv_feeder_unique_id"
    ts_name = "data_collection_log_timestamp"
    target_name = "mean_consumption_per_meter"

    lf = pl.scan_parquet(file)

    if n_series:
        ids = lf.select(group_id).unique().head(n_series)
        lf = lf.join(ids, on=group_id)

    lf = prepare_features(lf, group_id, ts_name, target_name)

    drop_cols = [ts_name, "temperature", "humidity", "aggregated_device_count_active"]
    if not return_unique_id:
        drop_cols.append(group_id)

    df = lf.drop(drop_cols).drop_nulls().collect()
    print(df.schema)

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

        base_callbacks = list(kwargs.pop("callbacks", []))

        for q in self.quantiles:
            with mlflow.start_run(nested=True, run_name=f"quantile_{q}"):
                current_params = {**self.params, "objective": "quantile", "alpha": q}
                mlflow.log_params(current_params)

                def mlflow_callback(env):
                    metrics = {
                        f"{d}_q{q}-{n}": s for d, n, s, _ in env.evaluation_result_list
                    }
                    if metrics:
                        mlflow.log_metrics(metrics, step=env.iteration)

                model = lgb.train(
                    current_params,
                    train_set=train_data,
                    valid_sets=[val_data],
                    callbacks=base_callbacks + [mlflow_callback],
                    **kwargs,
                )
                q_str = str(q).replace(".", "_")
                mlflow.lightgbm.log_model(model, name=f"model_q{q_str}")
                ax = lgb.plot_importance(model, importance_type="gain")
                plt.tight_layout()
                mlflow.log_figure(ax.figure, "feature_importance.png")
                self.models[q] = model
        return self

    def predict(self, X):
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        return preds


def main():
    path = Path("dataset")
    X_train, y_train = prepare_dataset(path / "train.parquet")
    X_val, y_val = prepare_dataset(path / "val.parquet")
    X_test, y_test = prepare_dataset(path / "test.parquet")

    quantiles = [0.1, 0.5, 0.9]

    model = MultiQuantileLGBM(
        quantiles=quantiles,
        force_row_wise=True,
        data_sample_strategy="goss",
        n_estimators=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run():
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(1), lgb.log_evaluation(1)],
        )
        preds = model.predict(X_test)

        pinball_losses = []
        for i, q in enumerate(quantiles):
            loss = mean_pinball_loss(y_test, preds[:, i], alpha=q)
            pinball_losses.append(loss)
            mlflow.log_metric(f"test_pinball_loss_q{q}", loss)

        mlflow.log_metric("test_pinball_loss_avg", np.mean(pinball_losses))

    joblib.dump(model, "lgb_model.pkl")


if __name__ == "__main__":
    main()
