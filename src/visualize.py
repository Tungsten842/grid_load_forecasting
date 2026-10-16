import warnings
from pathlib import Path

import joblib
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from train_tree import MultiQuantileLGBM, prepare_dataset  # noqa

warnings.filterwarnings("ignore")


def build_forecast_plot(y_true, y_preds, quantiles, median_idx, title):
    fig = go.Figure()
    n_quantiles = len(quantiles)

    if n_quantiles >= 2:
        fig.add_trace(
            go.Scatter(
                y=y_preds[:, 0],
                name=f"q={quantiles[0]}",
                line={"color": "rgba(31, 119, 180, 0.4)", "width": 1, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                y=y_preds[:, -1],
                name=f"q={quantiles[-1]}",
                line={"color": "rgba(31, 119, 180, 0.4)", "width": 1, "dash": "dash"},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.2)",
            )
        )

    for q_i in range(1, n_quantiles - 1):
        if q_i != median_idx:
            fig.add_trace(
                go.Scatter(
                    y=y_preds[:, q_i],
                    name=f"q={quantiles[q_i]}",
                    line={"dash": "dot"},
                )
            )

    fig.add_trace(
        go.Scatter(
            y=y_preds[:, median_idx],
            name=f"Pred (q={quantiles[median_idx]})",
            line={"color": "blue", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            y=y_true,
            name="Actual",
            line={"color": "black", "width": 1.5},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Consumption",
    )
    return fig


def main():
    st.title("Test Forecast Evaluation")

    model = joblib.load("lgb_model.pkl")

    n_series = 40
    X, y = prepare_dataset(Path("dataset/test.parquet"), n_series, True)

    # Slice first week per time series
    df = (
        X.with_columns(_y=y)
        .group_by("lv_feeder_unique_id", maintain_order=True)
        .head(336)
        .with_row_index()
    )

    series_chunks = [
        idx.to_numpy()
        for idx in df.group_by("lv_feeder_unique_id", maintain_order=True).agg(
            pl.col("index")
        )["index"]
    ]

    X = df.drop("index", "_y", "lv_feeder_unique_id")
    y = df["_y"].to_numpy()

    quantiles = model.quantiles
    n_quantiles = len(quantiles)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else n_quantiles // 2

    preds = model.predict(X)

    # Plot each time series
    for i, idx in enumerate(series_chunks):
        fig = build_forecast_plot(
            y_true=y[idx],
            y_preds=preds[idx],
            quantiles=quantiles,
            median_idx=median_idx,
            title=f"Time Series {i + 1}",
        )
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
