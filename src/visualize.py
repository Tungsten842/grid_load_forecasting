import warnings
from pathlib import Path

import joblib
import plotly.graph_objects as go
import polars as pl
import streamlit as st
import torch

from train_nn import LoadForecaster
from train_tree import MultiQuantileLGBM, prepare_dataset  # noqa

warnings.filterwarnings("ignore")


def build_forecast_plot(
    y_true, history, lgb_pred, nn_pred, quantiles, median_idx, title
):
    fig = go.Figure()
    n_quantiles = len(quantiles)

    fig.add_trace(
        go.Scatter(
            x=list(range(-len(history), 0)),
            y=history,
            name="History (Past Week)",
            line={"color": "black", "width": 1.5},
        )
    )

    fig.add_trace(
        go.Scatter(
            y=lgb_pred[:, 0],
            name=f"q={quantiles[0]}",
            line={"color": "rgba(31, 119, 180, 0.4)", "width": 1, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            y=lgb_pred[:, -1],
            name=f"q={quantiles[-1]}",
            line={"color": "rgba(31, 119, 180, 0.4)", "width": 1, "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=nn_pred[:, 0],
            name=f"NN q={quantiles[0]}",
            line={
                "color": "rgba(214, 39, 40, 0.4)",
                "width": 1,
                "dash": "dash",
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            y=nn_pred[:, -1],
            name=f"NN q={quantiles[-1]}",
            line={
                "color": "rgba(214, 39, 40, 0.4)",
                "width": 1,
                "dash": "dash",
            },
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )
    for q_i in range(1, n_quantiles - 1):
        if q_i != median_idx:
            fig.add_trace(
                go.Scatter(
                    y=nn_pred[:, q_i],
                    name=f"NN q={quantiles[q_i]}",
                    line={"dash": "dot"},
                )
            )
    fig.add_trace(
        go.Scatter(
            y=nn_pred[:, median_idx],
            name=f"NN Pred (q={quantiles[median_idx]})",
            line={"color": "red", "width": 2},
        )
    )

    for q_i in range(1, n_quantiles - 1):
        if q_i != median_idx:
            fig.add_trace(
                go.Scatter(
                    y=lgb_pred[:, q_i],
                    name=f"q={quantiles[q_i]}",
                    line={"dash": "dot"},
                )
            )

    fig.add_trace(
        go.Scatter(
            y=lgb_pred[:, median_idx],
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

    fig.update_xaxes(
        range=[-96, len(y_true)],
        rangeslider={"visible": True},
    )
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("Test Forecast Evaluation")

    lgb_model = joblib.load("lgb_model.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = LoadForecaster().to(device)
    nn_model.load_state_dict(torch.load("model.pt"))
    nn_model.eval()

    n_series = 40
    X, y = prepare_dataset(Path("dataset/test.parquet"), n_series, True)

    full_df = X.with_columns(_y=y).filter(pl.len().over("lv_feeder_unique_id") >= 1008)

    nn_inputs = (
        full_df.group_by("lv_feeder_unique_id", maintain_order=True)
        .agg(pl.col("_y").cast(pl.Float32).head(672))
        .select(pl.col("_y").list.to_array(672))
        .to_series()
        .to_torch()
    )

    df = full_df.filter(
        pl.int_range(pl.len()).over("lv_feeder_unique_id").is_between(672, 1007)
    ).with_row_index()

    series_chunks = [
        idx.to_numpy()
        for idx in df.group_by("lv_feeder_unique_id", maintain_order=True).agg(
            pl.col("index")
        )["index"]
    ]

    X = df.drop("index", "_y", "lv_feeder_unique_id")
    y = df["_y"].to_numpy()

    quantiles = lgb_model.quantiles
    n_quantiles = len(quantiles)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else n_quantiles // 2

    preds = lgb_model.predict(X)

    with torch.inference_mode():
        nn_preds = nn_model(nn_inputs.to(device)).cpu().numpy()

    # Plot each time series
    for i, idx in enumerate(series_chunks):
        fig = build_forecast_plot(
            y_true=y[idx],
            history=nn_inputs[i],
            lgb_pred=preds[idx],
            nn_pred=nn_preds[i],
            quantiles=quantiles,
            median_idx=median_idx,
            title=f"Time Series {i + 1}",
        )
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
