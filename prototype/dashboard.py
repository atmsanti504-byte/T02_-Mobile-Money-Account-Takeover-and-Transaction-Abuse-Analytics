from pathlib import Path
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RISK_PATH = DATA_DIR / "ato_risk_scores.csv"
TIMELINE_PATH = DATA_DIR / "incident_timeline.csv"
SIMULATION_PATH = DATA_DIR / "simulation_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

risk = pd.read_csv(RISK_PATH)
timeline = pd.read_csv(TIMELINE_PATH)
simulation = pd.read_csv(SIMULATION_PATH)


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)

app.title = "ATO Fraud Detection Dashboard"


# ============================================================
# FIGURES
# ============================================================

risk_distribution = (
    risk["risk_level"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        fill_value=0
    )
    .reset_index()
)

risk_distribution.columns = [
    "risk_level",
    "count"
]

risk_fig = px.bar(
    risk_distribution,
    x="risk_level",
    y="count",
    title="ATO Risk Distribution",
    labels={
        "risk_level": "Risk Level",
        "count": "Transactions"
    }
)


simulation_fig = px.bar(
    simulation,
    x="scenario",
    y="mean_response_minutes",
    title="Simulated Mean Response Time",
    labels={
        "scenario": "Control Scenario",
        "mean_response_minutes": "Response Time (minutes)"
    }
)


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

app.layout = html.Div(

    style={
        "fontFamily": "Arial",
        "padding": "25px",
        "backgroundColor": "#f5f5f5"
    },

    children=[

        html.H1(
            "ATO Fraud Detection & Analyst Dashboard"
        ),

        html.P(
            "Mobile Money Account Takeover and "
            "Transaction Abuse Analytics"
        ),

        html.Hr(),

        # ====================================================
        # KPI CARDS
        # ====================================================

        html.Div(

            style={
                "display": "flex",
                "gap": "20px"
            },

            children=[

                html.Div(
                    [
                        html.H3("Transactions"),
                        html.H2(f"{len(risk):,}")
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "flex": "1"
                    }
                ),

                html.Div(
                    [
                        html.H3("Fraud Cases"),
                        html.H2(
                            f"{risk['is_fraud'].sum():,}"
                        )
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "flex": "1"
                    }
                ),

                html.Div(
                    [
                        html.H3("Critical Cases"),
                        html.H2(
                            f"{(risk['risk_level'] == 'CRITICAL').sum():,}"
                        )
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "flex": "1"
                    }
                ),

                html.Div(
                    [
                        html.H3("Mean Fraud Risk"),
                        html.H2(
                            f"{risk.loc[risk['is_fraud'] == 1, 'risk_score'].mean():.2f}"
                        )
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "flex": "1"
                    }
                )
            ]
        ),

        html.Br(),

        # ====================================================
        # RISK DISTRIBUTION
        # ====================================================

        dcc.Graph(
            figure=risk_fig
        ),

        # ====================================================
        # HIGH-RISK CASE QUEUE
        # ====================================================

        html.H2("Analyst Case Queue"),

        html.Div(
            [

                html.Label(
                    "Minimum Risk Score"
                ),

                dcc.Slider(
                    id="risk-slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.75,
                    marks={
                        0: "0",
                        0.25: "0.25",
                        0.5: "0.50",
                        0.75: "0.75",
                        1: "1"
                    }
                ),

            ],
            style={
                "backgroundColor": "white",
                "padding": "20px"
            }
        ),

        html.Br(),

        html.Div(
            id="case-table"
        ),

        html.Br(),

        # ====================================================
        # INCIDENT EVIDENCE
        # ====================================================

        html.H2(
            "Incident Evidence Timeline"
        ),

        html.Div(
            timeline.to_html(
                index=False,
                classes="timeline-table"
            ),
            style={
                "backgroundColor": "white",
                "padding": "15px",
                "overflowX": "auto"
            }
        ),

        html.Br(),

        # ====================================================
        # SIMULATION
        # ====================================================

        dcc.Graph(
            figure=simulation_fig
        ),

        html.Hr(),

        html.P(
            "Prototype developed using synthetic security data. "
            "Results are preliminary and intended for analyst "
            "decision support."
        )
    ]
)


# ============================================================
# CASE QUEUE CALLBACK
# ============================================================

@app.callback(
    Output("case-table", "children"),
    Input("risk-slider", "value")
)

def update_case_queue(minimum_score):

    cases = (
        risk[
            risk["risk_score"] >= minimum_score
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(20)
    )

    display_columns = [
        "transaction_id",
        "account_id",
        "xgb_probability",
        "anomaly_signal",
        "nlp_signal",
        "risk_score",
        "risk_level"
    ]

    cases = cases[display_columns].copy()

    for column in [
        "xgb_probability",
        "anomaly_signal",
        "nlp_signal",
        "risk_score"
    ]:
        cases[column] = cases[column].round(3)

    return html.Table(

        [

            html.Thead(
                html.Tr(
                    [
                        html.Th(column)
                        for column in cases.columns
                    ]
                )
            ),

            html.Tbody(

                [

                    html.Tr(
                        [
                            html.Td(row[column])
                            for column in cases.columns
                        ]
                    )

                    for _, row in cases.iterrows()

                ]

            )

        ],

        style={
            "width": "100%",
            "backgroundColor": "white",
            "borderCollapse": "collapse"})

if __name__ == "__main__":
    print("=" * 60)
    print("ATO ANALYST DASHBOARD")
    print("=" * 60)
    print("Starting dashboard...")
    app.run(debug=False)