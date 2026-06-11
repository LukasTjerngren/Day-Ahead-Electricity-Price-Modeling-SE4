"""Dash app around the model: an EDA tab, a feature-selection tab and a
model/evaluation tab. Run `python dashboard.py`, open http://127.0.0.1:8050."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table, Input, Output

import analysis as A
from hourly_dataset import TARGET_COLUMN, split_features_target
from hourly_model import run_holdout

# shared styling so the figures look like one app
TEMPLATE = "plotly_white"
BLUE, ORANGE, GREEN, GREY, INK = "#2563eb", "#f59e0b", "#10b981", "#cbd5e1", "#1e293b"
COLORWAY = [BLUE, ORANGE, GREEN, "#ef4444", "#8b5cf6"]
FONT = "Inter, system-ui, sans-serif"


def _style(fig, title=None, height=360):
    fig.update_layout(
        template=TEMPLATE, colorway=COLORWAY, height=height,
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=60, r=24, t=54 if title else 24, b=48),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=12)),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef2f7")
    return fig


# ---------------------------------------------------------------------------
# EDA figures
# ---------------------------------------------------------------------------

def fig_price_hist(df):
    fig = px.histogram(df, x=TARGET_COLUMN, nbins=100)
    fig.update_traces(marker_color=BLUE)
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.update_layout(bargap=0.02, xaxis_title="price (SEK/kWh)", yaxis_title="hours")
    return _style(fig, "Distribution of hourly prices")


def fig_daily_shape(profile):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profile["hour"], y=profile["q75"], line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=profile["hour"], y=profile["q25"], fill="tonexty",
                             fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
                             name="25–75%"))
    fig.add_trace(go.Scatter(x=profile["hour"], y=profile["mean"], name="mean",
                             line=dict(color=BLUE, width=3)))
    fig.update_layout(xaxis_title="local hour", yaxis_title="price (SEK/kWh)",
                      xaxis=dict(dtick=3))
    return _style(fig, "Average daily price shape")


def fig_price_vs_driver(df, col, label, color):
    """Binned mean price against a weather driver"""
    b = A.binned_price(df, col, bins=24)
    sample = df.sample(min(4000, len(df)), random_state=0)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=sample[col], y=sample[TARGET_COLUMN], mode="markers",
                               marker=dict(size=3, color=GREY, opacity=0.4),
                               name="hours", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=b["center"], y=b["mean"], mode="lines+markers",
                             line=dict(color=color, width=3), name="mean price"))
    fig.update_layout(xaxis_title=label, yaxis_title="price (SEK/kWh)",
                      yaxis_range=[-0.5, 2.5])
    return _style(fig)


def fig_resolution(res):
    labels = {24: "24 · hourly", 96: "96 · 15-min", 23: "23 · DST", 25: "25 · DST",
              92: "92 · DST+15min", 100: "100 · DST+15min"}
    x = [labels.get(i, str(i)) for i in res.index]
    fig = go.Figure(go.Bar(x=x, y=res.values, marker_color=BLUE,
                           text=res.values, textposition="outside"))
    fig.update_layout(xaxis_title="records per day", yaxis_title="number of days")
    return _style(fig, "Price feed resolution (data quality)")


# ---------------------------------------------------------------------------
# Feature-selection figures
# ---------------------------------------------------------------------------

def fig_corr_target(corr_t):
    s = corr_t[::-1]
    colors = [BLUE if v >= 0 else ORANGE for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                           text=[f"{v:+.2f}" for v in s.values], textposition="outside"))
    fig.update_layout(xaxis_title="Pearson r with tomorrow's price",
                      xaxis_range=[-1, 1])
    return _style(fig, "Linear correlation of each feature with the target", height=560)


def fig_corr_matrix(cmat):
    fig = px.imshow(cmat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return _style(fig, "Feature correlation matrix", height=620)


def fig_anova(anova):
    d = anova.sort_values("variance_explained")
    fig = go.Figure(go.Bar(
        x=d["variance_explained"] * 100, y=d["feature"], orientation="h", marker_color=BLUE,
        text=[f" {v*100:.1f}%" for v in d["variance_explained"]], textposition="outside",
        cliponaxis=False,
        customdata=np.c_[d["F_statistic"], d["p_value"]],
        hovertemplate="%{y}<br>explains %{x:.2f}% of price variance"
                      "<br>F=%{customdata[0]:.0f}, p=%{customdata[1]:.1e}<extra></extra>"))
    fig.update_layout(xaxis_title="% of price variance explained (eta²)",
                      xaxis_range=[0, d["variance_explained"].max() * 100 * 1.3])
    return _style(fig, "Price variance explained by each calendar split", height=360)


def fig_ablation(abl):
    """Drop-column ablation: holdout MAE rise per group, annotated with its max |r|."""
    d = abl[abl["group"] != "(full model)"].sort_values("delta_mae")
    colors = [ORANGE if g == "hour" else BLUE for g in d["group"]]
    fig = go.Figure(go.Bar(
        x=d["delta_mae"], y=d["group"], orientation="h", marker_color=colors,
        text=[f" |r|={r:.2f}" for r in d["max_abs_r"]], textposition="outside",
        cliponaxis=False))
    fig.update_layout(xaxis_title="holdout MAE rise when the group is dropped (SEK/kWh)",
                      xaxis_range=[0, d["delta_mae"].max() * 1.25])
    return _style(fig, "Drop-column ablation", height=560)


# ---------------------------------------------------------------------------
# Model / evaluation figures
# ---------------------------------------------------------------------------

def fig_importance(imp):
    s = imp[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=BLUE))
    fig.update_layout(xaxis_title="Δ MAE when shuffled (SEK/kWh)")
    return _style(fig, "Permutation feature importance", height=560)


def fig_pred_vs_actual(results, r2):
    sample = results.sample(min(3000, len(results)), random_state=0)
    lo, hi = results["actual"].min(), float(np.ceil(results["actual"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=sample["actual"], y=sample["predicted"], mode="markers",
                               marker=dict(size=4, color=BLUE, opacity=0.3), name="hours"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=INK, dash="dash"), name="perfect"))
    fig.update_layout(xaxis_title="actual (SEK/kWh)", yaxis_title="predicted (SEK/kWh)")
    return _style(fig, f"Predicted vs. actual  ·  R² = {r2:.3f}")


def fig_per_hour(phm):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=phm["hour"], y=phm["base"], name="baseline", marker_color=GREY))
    fig.add_trace(go.Bar(x=phm["hour"], y=phm["model"], name="model", marker_color=BLUE))
    fig.update_layout(barmode="group", xaxis_title="local hour", yaxis_title="MAE (SEK/kWh)",
                      xaxis=dict(dtick=2))
    return _style(fig, "Error by hour: model vs. persistence baseline")


def fig_residuals(results):
    resid = results["predicted"] - results["actual"]
    fig = px.histogram(resid, nbins=80)
    fig.update_traces(marker_color=BLUE)
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.update_layout(showlegend=False, xaxis_title="residual: predicted − actual (SEK/kWh)",
                      yaxis_title="hours")
    return _style(fig, "Residual distribution")


def fig_margin(margin):
    """Histogram of the per-hour margin over the baseline (analysis.margin_vs_baseline)."""
    fig = px.histogram(margin, nbins=80)
    fig.update_traces(marker_color=BLUE)
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.add_vline(x=float(margin.mean()), line_color=GREEN,
                  annotation_text=f"mean {margin.mean():+.3f}",
                  annotation_position="top right")
    fig.update_layout(showlegend=False,
                      xaxis_title="|baseline error| − |model error| (SEK/kWh)",
                      yaxis_title="hours")
    return _style(fig, "Margin over the baseline, hour by hour", height=320)


def fig_timeseries(results, month):
    sub = results[results["timestamp"].dt.strftime("%Y-%m") == month]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["baseline"], name="baseline",
                             line=dict(color=GREY, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["actual"], name="actual",
                             line=dict(color=INK, width=1.8)))
    fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["predicted"], name="model",
                             line=dict(color=BLUE, width=1.5)))
    fig.update_layout(xaxis_title=None, yaxis_title="price (SEK/kWh)")
    return _style(fig, f"Hourly price: actual vs. predicted · {month}", height=380)


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

def kpi(label, value, sub, sub_color=INK):
    return dbc.Card(dbc.CardBody([
        html.Div(label, style={"fontSize": 13, "color": "#64748b", "textTransform": "uppercase",
                               "letterSpacing": "0.04em"}),
        html.Div(value, style={"fontSize": 30, "fontWeight": 700, "color": INK, "lineHeight": "1.1"}),
        html.Div(sub, style={"fontSize": 13, "color": sub_color, "marginTop": "2px"}),
    ]), className="shadow-sm h-100")


def graph(fig):
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def note(text):
    """A small 'how to read this' caption shown under a panel title."""
    return dcc.Markdown(text, className="text-muted", style={"fontSize": 13, "marginBottom": "6px"})


def panel(title, body):
    return dbc.Card([
        dbc.CardHeader(title, style={"fontWeight": 600, "background": "white", "border": "none"}),
        dbc.CardBody(body),
    ], className="shadow-sm mb-4")


def _table(frame, decimals=3):
    rounded = frame.round(decimals)
    return dash_table.DataTable(
        data=rounded.to_dict("records"),
        columns=[{"name": c, "id": c} for c in rounded.columns],
        style_as_list_view=True,
        style_cell={"fontFamily": FONT, "fontSize": 13, "padding": "8px 14px", "border": "none"},
        style_header={"backgroundColor": "#f8fafc", "fontWeight": "600", "border": "none"},
        style_data={"borderBottom": "1px solid #eef2f7"},
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(holdout_months: int = 4) -> Dash:
    print("Building dataset and training model for the dashboard ...")
    h = run_holdout(holdout_months=holdout_months, verbose=True)
    df, results, model = h["df"], h["results"], h["model"]
    X_full, y_full = split_features_target(df)

    summary = (A.price_summary(df).rename("value").reset_index()
               .rename(columns={"index": "statistic"}))
    profile = A.hourly_profile(df)
    res = A.resolution_breakdown()
    corr_t = A.feature_target_correlations(X_full, y_full)
    cmat = A.correlation_matrix(X_full)
    anova = A.anova_by_group(df)
    abl = A.ablation_study(df)
    phm = A.per_hour_mae(results)
    imp = A.permutation_importances(model, h["X_test"], h["y_test"])
    m_model = A.evaluation_metrics(results["actual"], results["predicted"])
    m_base = A.evaluation_metrics(results["actual"], results["baseline"])

    mae_gain = (1 - m_model["MAE"] / m_base["MAE"]) * 100
    rmse_gain = (1 - m_model["RMSE"] / m_base["RMSE"]) * 100
    margin = A.margin_vs_baseline(results)
    win_rate = (margin > 0).mean() * 100
    scores = pd.DataFrame([
        {"": "Hourly model", "MAE": m_model["MAE"], "RMSE": m_model["RMSE"],
         "R²": m_model["R2"], "bias": m_model["bias"]},
        {"": "Persistence baseline", "MAE": m_base["MAE"], "RMSE": m_base["RMSE"],
         "R²": m_base["R2"], "bias": m_base["bias"]},
    ])
    months = sorted(results["timestamp"].dt.strftime("%Y-%m").unique())

    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
               title="SE4 hourly price model")

    def row(*cols):
        return dbc.Row([dbc.Col(c, md=w) for c, w in cols], className="g-4")

    # ---- Tab 1: EDA ----
    tab_eda = html.Div([
        html.P("How the price behaves, and how it moves with wind and temperature.",
               className="text-muted"),
        row((panel("Price distribution", graph(fig_price_hist(df))), 6),
            (panel("Average daily shape", graph(fig_daily_shape(profile))), 6)),
        row((panel("Price vs. wind  (supply driver)",
                   graph(fig_price_vs_driver(df, "wind_tomorrow_hour", "wind speed (m/s)", BLUE))), 6),
            (panel("Price vs. temperature  (checked, not a model input)",
                   graph(fig_price_vs_driver(df, "temp_tomorrow_hour", "temperature (°C)", ORANGE))), 6)),
        row((panel("Records per day", graph(fig_resolution(res))), 6),
            (panel("Target summary statistics", _table(summary)), 6)),
    ])

    # ---- Tab 2: feature selection ----
    reconcile = dcc.Markdown(
        "**Why keep features with near-zero linear correlation?** "
        "Pearson r only measures straight-line, one-feature-at-a-time "
        "relationships, and a boosted tree picks up non-linear and interaction "
        "effects that r is blind to. The better test is the right-hand chart: "
        "drop a feature group, retrain, remeasure the holdout error. "
        "Hour-of-day is the clearest case - |r| ≈ 0.10, but dropping it "
        "clearly hurts, because the daily curve is cyclical and a straight "
        "line can't fit that. By this test, no group left in the model is "
        "dead weight - temperature failed exactly this test across seasons "
        "(small winter gain, larger summer cost) and was cut. "
        "*(Season looks flat here only because the held-out window covers part "
        "of a year; over full years it does carry signal.)*",
        className="text-muted",
    )
    tab_features = html.Div([
        reconcile,
        html.Div(className="mb-3"),
        row((panel("Pearson r with the target", graph(fig_corr_target(corr_t))), 6),
            (panel("Holdout error when a group is dropped", graph(fig_ablation(abl))), 6)),
        row((panel("Calendar effects (ANOVA)", [
                note("Splits the price by each calendar variable and shows how much "
                     "of the variance the split explains (eta²). Every split is "
                     "'significant' - p ≈ 0 is inevitable with 40k rows - so bar "
                     "length is what matters. Hover for F and p."),
                graph(fig_anova(anova))]), 5),
            (panel("Feature correlation matrix", [
                note("Pairwise Pearson r between the features; bright red/blue blocks "
                     "flag redundant pairs (e.g. each weather variable with its delta)."),
                graph(fig_corr_matrix(cmat))]), 7)),
    ])

    # ---- Tab 3: model & evaluation ----
    kpis = row(
        (kpi("MAE", f"{m_model['MAE']:.3f}", f"▼ {mae_gain:.0f}% vs baseline", GREEN), 3),
        (kpi("RMSE", f"{m_model['RMSE']:.3f}", f"▼ {rmse_gain:.0f}% vs baseline", GREEN), 3),
        (kpi("R²", f"{m_model['R2']:.3f}", f"baseline {m_base['R2']:.3f}", INK), 3),
        (kpi("Beats baseline", f"{win_rate:.0f}%",
             f"of hours · avg {margin.mean():+.2f} SEK/kWh", INK), 3),
    )
    tab_model = html.Div([
        html.P(f"Out-of-sample on the last {holdout_months} months the model never "
               "saw, vs. the persistence baseline (tomorrow's hour = today's hour).",
               className="text-muted"),
        kpis,
        html.Div(className="mb-4"),
        panel("Scoreboard", _table(scores, decimals=4)),
        row((panel("Feature importance (held-out)", graph(fig_importance(imp))), 6),
            (panel("Predicted vs. actual", graph(fig_pred_vs_actual(results, m_model["R2"]))), 6)),
        row((panel("Error by hour", graph(fig_per_hour(phm))), 6),
            (panel("Residuals", graph(fig_residuals(results))), 6)),
        panel("Margin over the baseline", [
            note("|baseline error| − |model error| per held-out hour; right of "
                 "the red line the model was closer. The mean (green) equals "
                 "baseline MAE minus model MAE, the *by how much on average* "
                 "that the win-rate card alone doesn't say. A model that won "
                 "often by a little but lost rarely and big would sit left of "
                 "zero here despite a high win rate."),
            graph(fig_margin(margin)),
        ]),
        panel("Forecast window", [
            dcc.Dropdown(id="month", options=[{"label": m, "value": m} for m in months],
                         value=months[0], clearable=False,
                         style={"width": "200px", "marginBottom": "10px"}),
            dcc.Graph(id="timeseries", config={"displayModeBar": False}),
        ]),
    ])

    header = html.Div([
        html.H2("SE4 day-ahead hourly price model", className="mb-1"),
        html.P("Predicting tomorrow's 24 hourly spot prices from today's prices, "
               "SMHI wind forecasts, and the calendar.",
               className="text-muted mb-0"),
    ], className="py-4")

    app.layout = dbc.Container([
        header,
        dbc.Tabs(active_tab="eda", children=[
            dbc.Tab(tab_eda, tab_id="eda", label="1 · Exploratory analysis"),
            dbc.Tab(tab_features, tab_id="features", label="2 · Feature selection"),
            dbc.Tab(tab_model, tab_id="model", label="3 · Model & evaluation"),
        ]),
        html.Div(className="py-4"),
    ], fluid=False, style={"maxWidth": "1180px", "fontFamily": FONT})

    @app.callback(Output("timeseries", "figure"), Input("month", "value"))
    def _update(month):
        return fig_timeseries(results, month)

    return app


if __name__ == "__main__":
    create_app().run(debug=False, port=8050)
