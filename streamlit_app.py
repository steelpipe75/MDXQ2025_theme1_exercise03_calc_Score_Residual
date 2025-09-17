import pandas as pd
from st_aggrid import AgGrid
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib_fontja
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import altair as alt
import io
import os


SALES_HISTORY_PATH = "./data/sales_history.csv"
ITEM_CATEGORIES_PATH = "./data/item_categories.csv"
CATEGORY_NAMES_PATH = "./data/category_names.csv"
TEST_PATH = "./data/test.csv"

ANSEWR_FILE_PATH = "./data/ans.csv"

# ---- ---- ---- ----

DEV_TYPE = "開発モード"
NOMAL_TYPE = "通常モード"

# ---- ---- ---- ----

RADIO_EVAL_NONE = "評価値/残差 表示しない"
RADIO_EVAL_TABLE = "評価値/残差 データ表(Streamlit標準 表形式)"
RADIO_EVAL_DF = "評価値/残差 データ表(Streamlit標準 DataFrame形式)"

RADIO_DETAIL_NONE = "詳細データ 表示しない"
RADIO_DETAIL_TABLE = "詳細データ(Streamlit標準 表形式)"
RADIO_DETAIL_DF = "詳細データ(Streamlit標準 DataFrame形式)"
RADIO_DETAIL_AGGRID = "詳細データ(AgGrid)"

RADIO_GRAPH_NONE = "グラフ 表示しない"
RADIO_MATPLOTLIB = "グラフ(matplotlib)"
RADIO_ALTAIR = "グラフ(Altair)"
RADIO_PLOTLY = "グラフ(Plotly)"
RADIO_BOKEH = "グラフ(Bokeh)"

RADIO_SHOP_ID = "店舗ID別"
RADIO_CATEGORY = "商品カテゴリ名別"
RADIO_SALES_RECORD = "前年販売実績あり別"

# ---- ---- ---- ----


def calc_score(y: pd.Series, y_hat: pd.Series):
    # MAEの計算
    mae = mean_absolute_error(y, y_hat)
    # RMSEの計算
    rmse = np.sqrt(mean_squared_error(y, y_hat))

    # RMSLEの計算
    # yとy_hatに負の値が含まれないことを確認
    if (y < 0).any() or (y_hat < 0).any():
        st.error(f"RMSLEは負の値を含むデータには適用できません")
        rmsle = np.nan
    else:
        rmsle = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(y_hat)))

    return mae, rmse, rmsle


def calculate_statistics(df: pd.DataFrame, column_name: str) -> pd.Series:
    statistics = df[column_name].describe()  # 基本統計量
    statistics["var"] = df[column_name].var()  # 分散

    return statistics.to_dict()


def score_series_helper(statistics, mae, rmse, rmsle, name):
    statistics["mae"] = mae
    statistics["rmse"] = rmse
    statistics["rmsle"] = rmsle
    df_score = pd.Series(data=statistics, name=name)

    df_score_rename = df_score.rename(
        {
            "mae": "評価値 全データで計算したMAE(参考)",
            "rmse": "評価値 全データで計算したRMSE",
            "rmsle": "評価値 全データで計算したRMSLE(参考)",
            "mean": "残差 平均値",
            "std": "残差 標準偏差",
            "var": "残差 分散",
            "min": "残差 最小値",
            "max": "残差 最大値",
            "25%": "残差 第1四分位数(25%)",
            "50%": "残差 中央値(第2四分位数,50%)",
            "75%": "残差 第3四分位数(75%)",
        }
    )
    df_score_drop = df_score_rename.drop("count")

    df_score_reindex = df_score_drop.reindex(
        index=[
            "評価値 全データで計算したRMSE",
            "評価値 全データで計算したMAE(参考)",
            "評価値 全データで計算したRMSLE(参考)",
            "残差 平均値",
            "残差 標準偏差",
            "残差 分散",
            "残差 最小値",
            "残差 第1四分位数(25%)",
            "残差 中央値(第2四分位数,50%)",
            "残差 第3四分位数(75%)",
            "残差 最大値",
        ]
    )

    return df_score_reindex


# ---- ---- ---- ----


def matplotlib_helper_sub(y_pred, y_true):
    # 散布図を作成
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, c="blue", s=64, label=f"正解 vs 予測", alpha=0.7)

    # データの範囲を計算
    min_value = min(min(y_pred), min(y_true))
    max_value = max(max(y_pred), max(y_true))

    # Y = X の直線を作成
    plt.plot([min_value, max_value], [min_value, max_value], "r--", label="Y = X")

    # グラフのタイトルと軸ラベルを設定
    plt.title(f"正解 vs 予測 Scatter Plot")
    plt.xlabel("正解")
    plt.ylabel("予測")

    # 軸の範囲を設定
    plt.xlim([min_value, max_value])
    plt.ylim([min_value, max_value])

    # 凡例を追加
    plt.legend()

    # グリッドを追加
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return plt


def sns_hist_helper(df, column_name: str):
    plt.figure(figsize=(7, 6))
    sns.histplot(df[column_name], kde=True, bins=10)
    plt.title(f"残差 ヒストグラム & KDE", fontsize=14)
    plt.xlabel(f"残差", fontsize=12)
    plt.ylabel("頻度", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    return plt


# ---- ---- ---- ----


def altair_helper_sub_by_shop_id(df):
    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # 凡例での選択を有効にする
    selection = alt.selection_point(fields=["店舗ID"], bind="legend")

    # 散布図を作成
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("正解", scale=alt.Scale(domain=[min_value, max_value]), title="正解"),
            y=alt.Y("予測", scale=alt.Scale(domain=[min_value, max_value]), title="予測"),
            color=alt.Color("店舗ID:N", legend=alt.Legend(title="店舗ID")),
            tooltip=["正解", "予測", "店舗ID", "商品ID", "商品カテゴリ名", "前年販売実績あり"],
        )
        .add_params(selection)
        .transform_filter(selection)
        .properties(title=f"正解 vs 予測 Scatter Plot (店舗ID別)", width=700, height=600)
        .interactive()
    )

    # Y = X の直線を作成
    line_data = pd.DataFrame({"x": [min_value, max_value], "y": [min_value, max_value]})
    line = (
        alt.Chart(line_data)
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x", y="y")
    )

    return scatter + line


def altair_helper_sub_by_category(df):
    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # 凡例での選択を有効にする
    selection = alt.selection_point(fields=["商品カテゴリ名"], bind="legend")

    # 散布図を作成
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("正解", scale=alt.Scale(domain=[min_value, max_value]), title="正解"),
            y=alt.Y("予測", scale=alt.Scale(domain=[min_value, max_value]), title="予測"),
            color=alt.Color("商品カテゴリ名:N", legend=alt.Legend(title="商品カテゴリ名")),
            tooltip=["正解", "予測", "店舗ID", "商品ID", "商品カテゴリ名", "前年販売実績あり"],
        )
        .add_params(selection)
        .transform_filter(selection)
        .properties(title=f"正解 vs 予測 Scatter Plot (商品カテゴリ名別)", width=700, height=600)
        .interactive()
    )

    # Y = X の直線を作成
    line_data = pd.DataFrame({"x": [min_value, max_value], "y": [min_value, max_value]})
    line = (
        alt.Chart(line_data)
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x", y="y")
    )

    return scatter + line


def altair_helper_sub_by_sales_record(df):
    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # 凡例での選択を有効にする
    selection = alt.selection_point(fields=["前年販売実績あり"], bind="legend")

    # 散布図を作成
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("正解", scale=alt.Scale(domain=[min_value, max_value]), title="正解"),
            y=alt.Y("予測", scale=alt.Scale(domain=[min_value, max_value]), title="予測"),
            color=alt.Color("前年販売実績あり:N", legend=alt.Legend(title="前年販売実績あり")),
            tooltip=["正解", "予測", "店舗ID", "商品ID", "商品カテゴリ名", "前年販売実績あり"],
        )
        .add_params(selection)
        .transform_filter(selection)
        .properties(title=f"正解 vs 予測 Scatter Plot (前年販売実績あり別)", width=700, height=600)
        .interactive()
    )

    # Y = X の直線を作成
    line_data = pd.DataFrame({"x": [min_value, max_value], "y": [min_value, max_value]})
    line = (
        alt.Chart(line_data)
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x", y="y")
    )

    return scatter + line


def altair_hist_helper(df, column_name):
    # データを抽出
    data = df[[column_name]].dropna()

    # ヒストグラムを作成
    hist = (
        alt.Chart(data)
        .mark_bar(opacity=0.5)
        .encode(
            alt.X(column_name, bin=alt.Bin(maxbins=30), title="残差"),
            alt.Y("count()", title="頻度"),
        )
    )

    # KDEを作成
    kde = (
        alt.Chart(data)
        .transform_density(
            column_name,
            as_=[column_name, "density"],
            extent=[data[column_name].min(), data[column_name].max()],
        )
        .mark_line()
        .encode(alt.X(column_name, title="残差"), alt.Y("density:Q", title="KDE"))
    )

    # ヒストグラムとKDEを重ねて表示し、二次軸を追加
    chart = (
        alt.layer(hist, kde)
        .resolve_scale(y="independent")
        .properties(title=f"残差 ヒストグラム & KDE")
        .interactive()
    )

    return chart


# ---- ---- ---- ----


def plotly_helper_sub_by_shop_id(df):
    # 図を作成
    fig = go.Figure()

    # 店舗IDごとに色を分けてプロット
    for store_id in df["店舗ID"].unique():
        store_df = df[df["店舗ID"] == store_id]
        fig.add_trace(
            go.Scatter(
                x=store_df["正解"],
                y=store_df["予測"],
                mode="markers",
                name=f"店舗ID {store_id}",
                marker=dict(size=8),
                customdata=store_df[["店舗ID", "商品ID", "商品カテゴリ名", "前年販売実績あり"]],
                hovertemplate="<b>正解: %{x}</b><br>"
                + "<b>予測: %{y}</b><br>"
                + "店舗ID: %{customdata[0]}<br>"
                + "商品ID: %{customdata[1]}<br>"
                + "商品カテゴリ名: %{customdata[2]}<br>"
                + "前年販売実績あり: %{customdata[3]}"
                + "<extra></extra>",
            )
        )

    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # Y = X の直線を作成
    line = go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="Y = X",
        line=dict(color="red", dash="dash"),
    )

    fig.add_trace(line)

    # レイアウト設定
    fig.update_layout(
        title=f"正解 vs 予測 Scatter Plot (店舗ID別)",
        xaxis_title="正解",
        yaxis_title="予測",
        showlegend=True,
        xaxis=dict(range=[min_value, max_value]),
        yaxis=dict(range=[min_value, max_value]),
        width=700,
        height=600,
    )

    return fig


def plotly_helper_sub_by_category(df):
    # 図を作成
    fig = go.Figure()

    # 商品カテゴリ名ごとに色を分けてプロット
    for category_name in df["商品カテゴリ名"].unique():
        category_df = df[df["商品カテゴリ名"] == category_name]
        fig.add_trace(
            go.Scatter(
                x=category_df["正解"],
                y=category_df["予測"],
                mode="markers",
                name=f"{category_name}",
                marker=dict(size=8),
                customdata=category_df[["店舗ID", "商品ID", "商品カテゴリ名", "前年販売実績あり"]],
                hovertemplate="<b>正解: %{x}</b><br>"
                + "<b>予測: %{y}</b><br>"
                + "店舗ID: %{customdata[0]}<br>"
                + "商品ID: %{customdata[1]}<br>"
                + "商品カテゴリ名: %{customdata[2]}<br>"
                + "前年販売実績あり: %{customdata[3]}"
                + "<extra></extra>",
            )
        )

    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # Y = X の直線を作成
    line = go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="Y = X",
        line=dict(color="red", dash="dash"),
    )

    fig.add_trace(line)

    # レイアウト設定
    fig.update_layout(
        title=f"正解 vs 予測 Scatter Plot (商品カテゴリ名別)",
        xaxis_title="正解",
        yaxis_title="予測",
        showlegend=True,
        xaxis=dict(range=[min_value, max_value]),
        yaxis=dict(range=[min_value, max_value]),
        width=700,
        height=600,
    )

    return fig


def plotly_helper_sub_by_sales_record(df):
    # 図を作成
    fig = go.Figure()

    # 前年販売実績ありごとに色を分けてプロット
    for sales_record in df["前年販売実績あり"].unique():
        record_df = df[df["前年販売実績あり"] == sales_record]
        fig.add_trace(
            go.Scatter(
                x=record_df["正解"],
                y=record_df["予測"],
                mode="markers",
                name=f"前年販売実績あり {sales_record}",
                marker=dict(size=8),
                customdata=record_df[["店舗ID", "商品ID", "商品カテゴリ名", "前年販売実績あり"]],
                hovertemplate="<b>正解: %{x}</b><br>"
                + "<b>予測: %{y}</b><br>"
                + "店舗ID: %{customdata[0]}<br>"
                + "商品ID: %{customdata[1]}<br>"
                + "商品カテゴリ名: %{customdata[2]}<br>"
                + "前年販売実績あり: %{customdata[3]}"
                + "<extra></extra>",
            )
        )

    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # Y = X の直線を作成
    line = go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="Y = X",
        line=dict(color="red", dash="dash"),
    )

    fig.add_trace(line)

    # レイアウト設定
    fig.update_layout(
        title=f"正解 vs 予測 Scatter Plot (前年販売実績あり別)",
        xaxis_title="正解",
        yaxis_title="予測",
        showlegend=True,
        xaxis=dict(range=[min_value, max_value]),
        yaxis=dict(range=[min_value, max_value]),
        width=700,
        height=600,
    )

    return fig


def plotly_hist_helper(df, column_name: str):
    # 指定された列からデータを抽出
    data = df[column_name].dropna().values
    # ヒストグラムとKDEを含むdistplotを作成
    fig = ff.create_distplot([data], group_labels=[column_name], bin_size=0.2)

    # レイアウトを更新してタイトルを追加
    fig.update_layout(
        title=f"残差 ヒストグラム & KDE)",
        xaxis_title="残差",
        yaxis_title="頻度",
    )

    return fig


# ---- ---- ---- ----


def bokeh_helper_sub_by_shop_id(df):
    # tooltipsを定義
    tooltips = [
        ("正解", "@x"),
        ("予測", "@y"),
        ("店舗ID", "@store_id"),
        ("商品ID", "@item_id"),
        ("商品カテゴリ名", "@category_name"),
        ("前年販売実績あり", "@sales_record"),
    ]

    # 散布図を作成
    p = figure(
        title=f"正解 vs 予測 Scatter Plot (店舗ID別)",
        x_axis_label="正解",
        y_axis_label="予測",
        width=700,
        height=600,
        tooltips=tooltips,
    )

    # 店舗IDごとに色を分けてプロット
    store_ids = df["店舗ID"].unique()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    import itertools

    color_cycle = itertools.cycle(colors)

    for store_id, color in zip(store_ids, color_cycle):
        store_df = df[df["店舗ID"] == store_id]
        source = ColumnDataSource(
            data=dict(
                x=store_df["正解"],
                y=store_df["予測"],
                store_id=store_df["店舗ID"],
                item_id=store_df["商品ID"],
                category_name=store_df["商品カテゴリ名"],
                sales_record=store_df["前年販売実績あり"],
            )
        )
        p.circle(
            "x",
            "y",
            source=source,
            size=8,
            color=color,
            alpha=0.7,
            legend_label=f"店舗ID {store_id}",
        )

    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # Y = X の直線を追加
    p.line(
        [min_value, max_value],
        [min_value, max_value],
        line_dash="dashed",
        line_color="red",
        legend_label="Y = X",
    )

    # 凡例の位置を設定
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def bokeh_helper_sub_by_category(df):
    # tooltipsを定義
    tooltips = [
        ("正解", "@x"),
        ("予測", "@y"),
        ("店舗ID", "@store_id"),
        ("商品ID", "@item_id"),
        ("商品カテゴリ名", "@category_name"),
        ("前年販売実績あり", "@sales_record"),
    ]

    # 散布図を作成
    p = figure(
        title=f"正解 vs 予測 Scatter Plot (商品カテゴリ名別)",
        x_axis_label="正解",
        y_axis_label="予測",
        width=700,
        height=600,
        tooltips=tooltips,
    )

    # 商品カテゴリ名ごとに色を分けてプロット
    category_names = df["商品カテゴリ名"].unique()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    import itertools

    color_cycle = itertools.cycle(colors)

    for category_name, color in zip(category_names, color_cycle):
        category_df = df[df["商品カテゴリ名"] == category_name]
        source = ColumnDataSource(
            data=dict(
                x=category_df["正解"],
                y=category_df["予測"],
                store_id=category_df["店舗ID"],
                item_id=category_df["商品ID"],
                category_name=category_df["商品カテゴリ名"],
                sales_record=category_df["前年販売実績あり"],
            )
        )
        p.circle(
            "x",
            "y",
            source=source,
            size=8,
            color=color,
            alpha=0.7,
            legend_label=f"{category_name}",
        )

    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # Y = X の直線を追加
    p.line(
        [min_value, max_value],
        [min_value, max_value],
        line_dash="dashed",
        line_color="red",
        legend_label="Y = X",
    )

    # 凡例の位置を設定
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def bokeh_helper_sub_by_sales_record(df):
    # tooltipsを定義
    tooltips = [
        ("正解", "@x"),
        ("予測", "@y"),
        ("店舗ID", "@store_id"),
        ("商品ID", "@item_id"),
        ("商品カテゴリ名", "@category_name"),
        ("前年販売実績あり", "@sales_record"),
    ]

    # 散布図を作成
    p = figure(
        title=f"正解 vs 予測 Scatter Plot (前年販売実績あり別)",
        x_axis_label="正解",
        y_axis_label="予測",
        width=700,
        height=600,
        tooltips=tooltips,
    )

    # 前年販売実績ありごとに色を分けてプロット
    sales_records = df["前年販売実績あり"].unique()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    import itertools

    color_cycle = itertools.cycle(colors)

    for sales_record, color in zip(sales_records, color_cycle):
        record_df = df[df["前年販売実績あり"] == sales_record]
        source = ColumnDataSource(
            data=dict(
                x=record_df["正解"],
                y=record_df["予測"],
                store_id=record_df["店舗ID"],
                item_id=record_df["商品ID"],
                category_name=record_df["商品カテゴリ名"],
                sales_record=record_df["前年販売実績あり"],
            )
        )
        p.circle(
            "x",
            "y",
            source=source,
            size=8,
            color=color,
            alpha=0.7,
            legend_label=f"前年販売実績あり {sales_record}",
        )

    # データの範囲を計算
    min_value = min(min(df["予測"]), min(df["正解"]))
    max_value = max(max(df["予測"]), max(df["正解"]))

    # Y = X の直線を追加
    p.line(
        [min_value, max_value],
        [min_value, max_value],
        line_dash="dashed",
        line_color="red",
        legend_label="Y = X",
    )

    # 凡例の位置を設定
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def bokeh_hist_helper(df, column_name: str):
    # データを抽出
    data = df[column_name].dropna().values

    # ヒストグラムを作成
    hist, edges = np.histogram(data, bins=30, density=True)

    # Bokehプロットを作成
    p_hist = figure(
        title=f"残差 ヒストグラム",
        x_axis_label="残差",
        y_axis_label="頻度",
        width=800,
        height=400,
    )

    # ヒストグラムを追加
    p_hist.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_alpha=0.5,
        line_color="black",
    )

    return p_hist


# ---- ---- ---- ----


def app(dev_mode):
    data_expander = st.expander("データ登録", expanded=True)

    st.divider()

    with data_expander:
        if not dev_mode:
            col_0_0, col_0_1, col_0_2 = st.columns(3)
            with col_0_0:
                st.subheader("販売実績データ")
                sales_history_file = st.file_uploader(
                    "Competitionサイトで公開された販売実績データファイル(sales_history.csv)をアップロード",
                    type="csv",
                )

            with col_0_1:
                st.subheader("商品カテゴリーデータ")
                item_categories_file = st.file_uploader(
                    "Competitionサイトで公開された商品カテゴリーデータファイル(item_categories.csv)をアップロード",
                    type="csv",
                )

            with col_0_2:
                st.subheader("カテゴリ名称データ")
                category_names_file = st.file_uploader(
                    "Competitionサイトで公開されたカテゴリ名称データファイル(category_names.csv)をアップロード",
                    type="csv",
                )

            col_1_0, col_1_1, col_1_2 = st.columns(3)
            with col_1_0:
                st.subheader("評価用データ")
                test_file = st.file_uploader(
                    "Competitionサイトで公開された評価用データ(test.csv)をアップロード",
                    type="csv",
                )

            with col_1_1:
                st.subheader("正解データ")
                answer_file = st.file_uploader(
                    "Competitionサイトで公開された正解データファイル(ans.csv)をアップロード",
                    type="csv",
                )

            with col_1_2:
                st.subheader("投稿データ")
                submit_file = st.file_uploader(
                    "投稿ファイル(csv)をアップロード", type="csv"
                )
        else:
            st.subheader("投稿データ")
            submit_file = st.file_uploader(
                "投稿ファイル(csv)をアップロード", type="csv"
            )

    if dev_mode:
        sales_history_df = pd.read_csv(SALES_HISTORY_PATH)
        item_categories_df = pd.read_csv(ITEM_CATEGORIES_PATH)
        category_names_df = pd.read_csv(CATEGORY_NAMES_PATH)
        test_df = pd.read_csv(TEST_PATH)

        anser_df = pd.read_csv(ANSEWR_FILE_PATH, header=None)
        anser_df.columns = ["index", "正解"]
    else:
        if sales_history_file is not None:
            sales_history_df = pd.read_csv(sales_history_file)
        if item_categories_file is not None:
            item_categories_df = pd.read_csv(item_categories_file)
        if category_names_file is not None:
            category_names_df = pd.read_csv(category_names_file)
        if test_file is not None:
            test_df = pd.read_csv(test_file)

        if answer_file is not None:
            anser_df = pd.read_csv(answer_file, header=None)
            anser_df.columns = ["index", "正解"]

    if submit_file is not None:
        submit_df = pd.read_csv(submit_file, header=None)
        submit_df.columns = ["index", "予測"]

    calc_enable = False

    if not dev_mode:
        if (
            sales_history_file is not None
            and item_categories_file is not None
            and category_names_file is not None
            and test_file is not None
            and answer_file is not None
            and submit_file is not None
        ):
            calc_enable = True
    else:
        if submit_file is not None:
            calc_enable = True

    if calc_enable:
        # ---- VVV 計算処理 VVV ----

        work1_df = pd.merge(test_df, anser_df)
        work2_df = pd.merge(work1_df, submit_df)
        work3_df = pd.merge(work2_df, item_categories_df, on="商品ID", how="left")
        merged_df = pd.merge(
            work3_df, category_names_df, on="商品カテゴリID", how="left"
        )

        merged_df["残差"] = merged_df["正解"] - merged_df["予測"]
        merged_df["残差^2"] = merged_df["残差"] * merged_df["残差"]

        mae, rmse, rmsle = calc_score(merged_df["正解"], merged_df["予測"])
        statistics_all = calculate_statistics(merged_df, "残差")
        score_all = score_series_helper(
            statistics_all, mae, rmse, rmsle, name="評価値/残差 データ"
        )

        # test_dfに存在する商品IDのうち、sales_history_dfの2021年の売り上げがない商品をリストアップ
        # '日付'カラムをdatetime型に変換
        sales_history_df['日付'] = pd.to_datetime(sales_history_df['日付'])

        # 2021年のデータのみを抽出
        sales_2021_df = sales_history_df[sales_history_df['日付'].dt.year == 2021]

        # 2021年に売上があった商品IDのセット
        sold_items_2021 = set(sales_2021_df['商品ID'].unique())

        # test_dfに存在する全ての商品IDのセット
        test_items = set(test_df['商品ID'].unique())

        # test_dfに存在するが、2021年に売上がなかった商品IDを特定
        items_no_sales_2021 = list(test_items - sold_items_2021)

        test_group_df = pd.DataFrame({'商品ID' : list(test_items)})
        test_group_df['前年販売実績あり'] = True
        test_group_df.loc[test_group_df['商品ID'].isin(items_no_sales_2021), '前年販売実績あり'] = False

        merged_df = pd.merge(merged_df, test_group_df, on=['商品ID'], how='left')

        # ---- AAA 計算処理 AAA ----

        dowunload_csv = "テーマ1 演習03 評価値/残差 計算\n"
        dowunload_csv += "\n"
        dowunload_csv += score_all.to_csv()
        dowunload_csv += "\n"
        dowunload_csv += "\n詳細データ\n"
        dowunload_csv += merged_df.to_csv()
        dowunload_csv += "\n"

        col_2_0, col_2_1 = st.columns(2)

        with col_2_0:
            st.metric(label="評価値 (全データで計算したRMSE)", value=score_all.loc["評価値 全データで計算したRMSE"])

        with col_2_1:
            st.download_button(
                label="CSVで計算したデータをダウンロード",
                data=dowunload_csv,
                file_name=f"rmse_{rmse}_theme01_exercise03_{submit_file.name}_score.csv",
                mime="text/csv",
            )

        eval_expander = st.expander("評価値/残差統計情報", expanded=False)
        detail_expander = st.expander("残差詳細情報", expanded=False)
        graph_matplotlib_expander = st.expander("グラフ(matplotlib)", expanded=False)
        graph_interactive_expander = st.expander("グラフ(インタラクティブ)", expanded=False)

        with eval_expander:
            select_eval_mode = st.radio(
                label="評価値/残差統計情報 モード選択",
                options=[
                    RADIO_EVAL_NONE,
                    RADIO_EVAL_TABLE,
                    RADIO_EVAL_DF,
                ],
                horizontal=True,
                label_visibility="hidden",
            )

            if select_eval_mode == RADIO_EVAL_TABLE:
                st.header("評価値/残差 データ表(Streamlit標準 表形式)")
                st.subheader("評価値/残差 データ")
                st.table(score_all)

            if select_eval_mode == RADIO_EVAL_DF:
                st.header("評価値/残差 データ表(Streamlit標準 DataFrame形式)")
                st.subheader("評価値/残差 データ")
                st.dataframe(score_all)

        with detail_expander:
            select_detail_mode = st.radio(
                label="残差詳細情報 モード選択",
                options=[
                    RADIO_DETAIL_NONE,
                    RADIO_DETAIL_TABLE,
                    RADIO_DETAIL_DF,
                    RADIO_DETAIL_AGGRID,
                ],
                horizontal=True,
                label_visibility="hidden",
            )

            if select_detail_mode == RADIO_DETAIL_TABLE:
                st.header("詳細データ(Streamlit標準 表形式)")
                st.table(merged_df)

            if select_detail_mode == RADIO_DETAIL_DF:
                st.header("詳細データ(Streamlit標準 DataFrame形式)")
                st.dataframe(merged_df)

            if select_detail_mode == RADIO_DETAIL_AGGRID:
                st.header("詳細データ(AgGrid)")
                AgGrid(merged_df, fit_columns_on_grid_load=True)

        with graph_matplotlib_expander:
            select_graph_matplotlib_mode = st.radio(
                label="モード選択",
                options=[
                    RADIO_GRAPH_NONE,
                    RADIO_MATPLOTLIB,
                ],
                horizontal=True,
                label_visibility="hidden",
            )

            if select_graph_matplotlib_mode == RADIO_MATPLOTLIB:
                st.header("グラフ(matplotlib)")
                st.subheader("正解 対 予測 グラフ")
                plt_all = matplotlib_helper_sub(merged_df["予測"], merged_df["正解"])
                st.pyplot(plt_all)
                plt_all.close()


        with graph_interactive_expander:
            select_graph_interactive_lib = st.radio(
                label="グラフライブラリ選択",
                options=[
                    RADIO_GRAPH_NONE,
                    RADIO_ALTAIR,
                    RADIO_PLOTLY,
                    RADIO_BOKEH,
                ],
                horizontal=True,
                label_visibility="hidden",
                key="interactive_lib",
            )
            select_graph_div_mode = st.radio(
                label="モード選択",
                options=[
                    RADIO_SHOP_ID,
                    RADIO_CATEGORY,
                    RADIO_SALES_RECORD,
                ],
                horizontal=True,
                label_visibility="hidden",
                key="graph_div_mode",
            )

            if select_graph_interactive_lib == RADIO_ALTAIR:
                st.header("グラフ(Altair)")
                if select_graph_div_mode == RADIO_SHOP_ID:
                    st.subheader("正解 対 予測 グラフ (店舗ID別)")
                    plt_shop_id = altair_helper_sub_by_shop_id(merged_df)
                    st.altair_chart(plt_shop_id)
                if select_graph_div_mode == RADIO_CATEGORY:
                    st.subheader("正解 対 予測 グラフ (商品カテゴリ名別)")
                    plt_cat = altair_helper_sub_by_category(merged_df)
                    st.altair_chart(plt_cat)
                if select_graph_div_mode == RADIO_SALES_RECORD:
                    st.subheader("正解 対 予測 グラフ (前年販売実績あり別)")
                    plt_sales = altair_helper_sub_by_sales_record(merged_df)
                    st.altair_chart(plt_sales)

                st.divider()
                st.subheader("残差 グラフ")
                st.altair_chart(altair_hist_helper(merged_df, "残差"))

            if select_graph_interactive_lib == RADIO_PLOTLY:
                st.header("グラフ(Plotly)")
                if select_graph_div_mode == RADIO_SHOP_ID:
                    st.subheader("正解 対 予測 グラフ (店舗ID別)")
                    plt_shop_id = plotly_helper_sub_by_shop_id(merged_df)
                    st.plotly_chart(plt_shop_id)
                if select_graph_div_mode == RADIO_CATEGORY:
                    st.subheader("正解 対 予測 グラフ (商品カテゴリ名別)")
                    plt_cat = plotly_helper_sub_by_category(merged_df)
                    st.plotly_chart(plt_cat)
                if select_graph_div_mode == RADIO_SALES_RECORD:
                    st.subheader("正解 対 予測 グラフ (前年販売実績あり別)")
                    plt_sales = plotly_helper_sub_by_sales_record(merged_df)
                    st.plotly_chart(plt_sales)

                st.divider()
                st.subheader("残差 グラフ")
                st.plotly_chart(plotly_hist_helper(merged_df, "残差"))

            if select_graph_interactive_lib == RADIO_BOKEH:
                st.header("グラフ(Bokeh)")
                if select_graph_div_mode == RADIO_SHOP_ID:
                    st.subheader("正解 対 予測 グラフ (店舗ID別)")
                    plt_shop_id = bokeh_helper_sub_by_shop_id(merged_df)
                    st.bokeh_chart(plt_shop_id)
                if select_graph_div_mode == RADIO_CATEGORY:
                    st.subheader("正解 対 予測 グラフ (商品カテゴリ名別)")
                    plt_cat = bokeh_helper_sub_by_category(merged_df)
                    st.bokeh_chart(plt_cat)
                if select_graph_div_mode == RADIO_SALES_RECORD:
                    st.subheader("正解 対 予測 グラフ (前年販売実績あり別)")
                    plt_sales = bokeh_helper_sub_by_sales_record(merged_df)
                    st.bokeh_chart(plt_sales)

                st.divider()
                st.subheader("残差 グラフ")
                st.bokeh_chart(bokeh_hist_helper(merged_df, "残差"))

    else:
        st.markdown("ファイルをアップロードするとここに評価値/残差を表示します")


# ---- ---- ---- ----

if __name__ == "__main__":
    st.set_page_config(
        page_title="MDXQ2025 テーマ1 演習03 評価値/残差 計算アプリ",
        layout="wide",
        page_icon=":computer:",
    )
    st.title("MDXQ2025 テーマ1 演習03 評価値/残差 計算アプリ")
    st.markdown("**Public / Private のどちらの計算にどのデータが使われたかが公開されていない**ため、**全データ**を使って評価値を計算しています。")

    dev_mode_enable = False
    if os.path.isfile(TEST_PATH):
        dev_mode_enable = True

    fix_dev_mode = False
    # fix_dev_mode = True

    dev_mode = False
    if dev_mode_enable:
        if not fix_dev_mode:
            select_dev_mode = st.radio(
                label="モード選択",
                options=[DEV_TYPE, NOMAL_TYPE],
                horizontal=True,
                label_visibility="hidden",
            )
            if select_dev_mode == DEV_TYPE:
                dev_mode = True
        else:
            dev_mode = True

    # dev_mode = False # 強制的に通常モードにする

    app(dev_mode)
