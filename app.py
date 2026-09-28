import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go

from energy_env import EnergyStorageEnv
from battery import Battery
from dueling_dqn import DuelingDQN
from double_dqn import DoubleDQN


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Energy Storage Arbitrage",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

EPISODE_LENGTH = 24

ACTION_NAMES = {
    0: "IDLE",
    1: "CHARGE",
    2: "DISCHARGE",
}

ACTION_COLORS = {
    "IDLE": "#64748B",
    "CHARGE": "#16A34A",
    "DISCHARGE": "#DC2626",
}

MODEL_FILES = {
    "Dueling DQN": "dueling_dqn_model.pth",
    "Double DQN": "double_dqn_model.pth",
}

MODEL_CLASSES = {
    "Dueling DQN": DuelingDQN,
    "Double DQN": DoubleDQN,
}


# ============================================================
# DATA / MODEL HELPERS
# ============================================================

@st.cache_data
def load_price_data(scenario):
    if scenario == "Day-Ahead Price":
        file_name = "real_test_prices.csv"
        column_name = "price day ahead"
    else:
        file_name = "real_test_actual_prices.csv"
        column_name = "price actual"

    df = pd.read_csv(file_name)

    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' was not found in {file_name}."
        )

    prices = df[column_name].astype(float).to_numpy()
    return prices, file_name


@st.cache_resource
def load_model(model_name):
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")

    model = MODEL_CLASSES[model_name](
        state_size=3,
        action_size=3
    )

    model.load_state_dict(
        torch.load(
            MODEL_FILES[model_name],
            map_location="cpu"
        )
    )

    model.eval()
    return model


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def load_training_rewards(model_name):
    file_name = (
        "dueling_dqn_training_rewards.txt"
        if model_name == "Dueling DQN"
        else "double_dqn_training_rewards.txt"
    )

    try:
        df = pd.read_csv(file_name)
    except Exception:
        return None

    if "Reward" not in df.columns:
        return None

    reward = pd.to_numeric(
        df["Reward"],
        errors="coerce"
    ).dropna()

    if reward.empty:
        return None

    result = pd.DataFrame({
        "Episode": np.arange(1, len(reward) + 1),
        "Reward": reward.to_numpy()
    })

    result["Moving Average"] = (
        result["Reward"]
        .rolling(window=100, min_periods=1)
        .mean()
    )

    return result


# ============================================================
# ENVIRONMENT / SIMULATION HELPERS
# ============================================================

def make_environment(prices, battery_capacity):
    env = EnergyStorageEnv(prices)

    env.battery = Battery(
        capacity_kwh=battery_capacity,
        initial_energy_kwh=battery_capacity * 0.50,
        min_soc=0.10,
        max_soc=0.90,
        max_charge_kw=20,
        max_discharge_kw=20,
        charge_efficiency=0.90,
        discharge_efficiency=0.90,
    )

    return env


def run_model(model, daily_prices, battery_capacity):
    env = make_environment(
        daily_prices,
        battery_capacity
    )

    state, _ = env.reset()

    records = []
    total_reward = 0.0

    charge_count = 0
    discharge_count = 0
    idle_count = 0

    for hour in range(EPISODE_LENGTH):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(
                q_values,
                dim=1
            ).item()

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += reward

        action_name = ACTION_NAMES[action]

        if action == 0:
            idle_count += 1
        elif action == 1:
            charge_count += 1
        else:
            discharge_count += 1

        records.append({
            "Hour": hour,
            "Price": float(info["price"]),
            "Action": action_name,
            "Action Number": action,
            "Reward": float(reward),
            "SOC": float(info["soc"] * 100),
        })

        state = next_state

        if terminated or truncated:
            break

    results = pd.DataFrame(records)

    return {
        "results": results,
        "total_reward": float(total_reward),
        "final_soc": float(results["SOC"].iloc[-1]),
        "charge": int(charge_count),
        "discharge": int(discharge_count),
        "idle": int(idle_count),
    }


def run_price_based(daily_prices, battery_capacity):
    env = make_environment(
        daily_prices,
        battery_capacity
    )

    low_price = np.percentile(daily_prices, 25)
    high_price = np.percentile(daily_prices, 75)

    records = []
    total_reward = 0.0

    charge_count = 0
    discharge_count = 0
    idle_count = 0

    for hour in range(EPISODE_LENGTH):

        price = daily_prices[hour]

        if price <= low_price:
            action = 1
        elif price >= high_price:
            action = 2
        else:
            action = 0

        (
            _,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += reward

        action_name = ACTION_NAMES[action]

        if action == 0:
            idle_count += 1
        elif action == 1:
            charge_count += 1
        else:
            discharge_count += 1

        records.append({
            "Hour": hour,
            "Price": float(info["price"]),
            "Action": action_name,
            "Action Number": action,
            "Reward": float(reward),
            "SOC": float(info["soc"] * 100),
        })

        if terminated or truncated:
            break

    results = pd.DataFrame(records)

    return {
        "results": results,
        "total_reward": float(total_reward),
        "final_soc": float(results["SOC"].iloc[-1]),
        "charge": charge_count,
        "discharge": discharge_count,
        "idle": idle_count,
    }


def run_no_battery(daily_prices):
    results = pd.DataFrame({
        "Hour": np.arange(len(daily_prices)),
        "Price": daily_prices.astype(float),
        "Action": ["IDLE"] * len(daily_prices),
        "Action Number": [0] * len(daily_prices),
        "Reward": [0.0] * len(daily_prices),
        "SOC": [0.0] * len(daily_prices),
    })

    return {
        "results": results,
        "total_reward": 0.0,
        "final_soc": 0.0,
        "charge": 0,
        "discharge": 0,
        "idle": len(daily_prices),
    }


# ============================================================
# PLOT HELPERS
# ============================================================

def make_price_figure(prices):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(prices)),
            y=prices,
            mode="lines+markers",
            name="Price",
            hovertemplate=(
                "Hour %{x}<br>"
                "Price: €%{y:.2f}/MWh"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Interactive Electricity Price Curve",
        xaxis_title="Hour",
        yaxis_title="Price (€/MWh)",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def make_price_action_figure(results):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["Hour"],
            y=results["Price"],
            mode="lines+markers",
            name="Electricity Price",
            line=dict(width=3),
            hovertemplate=(
                "Hour %{x}<br>"
                "Price: €%{y:.2f}/MWh"
                "<extra></extra>"
            ),
        )
    )

    for action_name in ["CHARGE", "DISCHARGE", "IDLE"]:

        subset = results[
            results["Action"] == action_name
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["Hour"],
                y=subset["Price"],
                mode="markers",
                name=action_name,
                marker=dict(
                    size=11,
                    color=ACTION_COLORS[action_name]
                ),
                customdata=subset[
                    ["SOC", "Reward"]
                ],
                hovertemplate=(
                    "Hour %{x}<br>"
                    "Price: €%{y:.2f}/MWh<br>"
                    f"Action: {action_name}<br>"
                    "SOC: %{customdata[0]:.1f}%<br>"
                    "Reward: €%{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Electricity Price vs Agent Decisions",
        xaxis_title="Hour",
        yaxis_title="Price (€/MWh)",
        hovermode="x unified",
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def make_soc_figure(first_results, second_results=None):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=first_results["Hour"],
            y=first_results["SOC"],
            mode="lines+markers",
            name="Dueling DQN" if second_results is not None else "SOC",
        )
    )

    if second_results is not None:
        fig.add_trace(
            go.Scatter(
                x=second_results["Hour"],
                y=second_results["SOC"],
                mode="lines+markers",
                name="Double DQN",
            )
        )

    fig.update_layout(
        title="Battery State of Charge",
        xaxis_title="Hour",
        yaxis_title="SOC (%)",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def make_reward_figure(rewards):
    fig = go.Figure(
        go.Bar(
            x=list(rewards.keys()),
            y=list(rewards.values()),
            text=[
                f"€{value:.2f}"
                for value in rewards.values()
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Reward: €%{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Reward Comparison",
        xaxis_title="Strategy",
        yaxis_title="Reward (€)",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def make_training_figure(model_names):
    fig = go.Figure()
    found = False

    for model_name in model_names:

        rewards = load_training_rewards(model_name)

        if rewards is None:
            continue

        found = True

        fig.add_trace(
            go.Scatter(
                x=rewards["Episode"],
                y=rewards["Reward"],
                mode="lines",
                name=f"{model_name} Reward",
                opacity=0.35,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rewards["Episode"],
                y=rewards["Moving Average"],
                mode="lines",
                name=f"{model_name} 100-Episode MA",
                line=dict(width=3),
            )
        )

    fig.update_layout(
        title="Training Reward Curves",
        xaxis_title="Episode",
        yaxis_title="Reward (€)",
        hovermode="x unified",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig, found


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Simulation Settings")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Live Simulation",
        "Model Comparison",
    ]
)

scenario = st.sidebar.selectbox(
    "Price Dataset",
    [
        "Day-Ahead Price",
        "Actual Price",
    ]
)

selected_model = None

if analysis_mode == "Live Simulation":
    selected_model = st.sidebar.selectbox(
        "Model Checkpoint",
        [
            "Dueling DQN",
            "Double DQN",
        ]
    )

test_day = st.sidebar.slider(
    "Select Test Day",
    min_value=1,
    max_value=292,
    value=1,
)

battery_capacity = st.sidebar.slider(
    "Battery Capacity (kWh)",
    min_value=50,
    max_value=200,
    value=100,
    step=10,
)

st.sidebar.caption(
    "The trained agents use a 100 kWh battery. "
    "Other capacities are sensitivity-analysis settings."
)


# ============================================================
# LOAD DATA
# ============================================================

try:
    prices, data_file = load_price_data(scenario)
except Exception as error:
    st.error(
        f"Could not load the selected dataset: {error}"
    )
    st.stop()

available_days = len(prices) // EPISODE_LENGTH

if test_day > available_days:
    st.error(
        f"Test day {test_day} is not available. "
        f"Available days: {available_days}"
    )
    st.stop()

start_index = (test_day - 1) * EPISODE_LENGTH
end_index = start_index + EPISODE_LENGTH
daily_prices = prices[start_index:end_index]


# ============================================================
# HEADER
# ============================================================

st.title("🔋 Energy Storage Arbitrage")

st.markdown(
    """
### Reinforcement Learning for Battery Energy Management

Interactive simulation of **Dueling DQN** and **Double DQN**
policies on real historical electricity-price data.
"""
)


# ============================================================
# TOP METRICS
# ============================================================

dueling_rewards = load_training_rewards("Dueling DQN")
double_rewards = load_training_rewards("Double DQN")

dueling_episodes = (
    len(dueling_rewards)
    if dueling_rewards is not None
    else 4000
)

double_episodes = (
    len(double_rewards)
    if double_rewards is not None
    else 4000
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Battery Capacity",
        f"{battery_capacity} kWh"
    )

with m2:
    st.metric(
        "Test Days",
        available_days
    )

with m3:
    st.metric(
        "Dueling DQN Episodes",
        dueling_episodes
    )

with m4:
    st.metric(
        "Double DQN Episodes",
        double_episodes
    )


# ============================================================
# TABS
# ============================================================

tab_live, tab_performance, tab_training, tab_architecture = st.tabs(
    [
        "⚡ Live Simulation",
        "📊 Model Performance",
        "📈 Training Analytics",
        "🧠 Architecture",
    ]
)


# ============================================================
# LIVE SIMULATION TAB
# ============================================================

with tab_live:

    st.subheader(
        f"24-Hour Price Profile — Test Day {test_day}"
    )

    st.plotly_chart(
        make_price_figure(daily_prices),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # LIVE MODE
    # --------------------------------------------------------

    if analysis_mode == "Live Simulation":

        run_button = st.button(
            f"▶ Run {selected_model} Simulation",
            type="primary",
            use_container_width=True,
        )

        if run_button:

            try:

                with st.spinner(
                    f"Running {selected_model}..."
                ):

                    model = load_model(selected_model)

                    simulation = run_model(
                        model,
                        daily_prices,
                        battery_capacity,
                    )

                results = simulation["results"]

                baseline = run_price_based(
                    daily_prices,
                    battery_capacity,
                )

                st.markdown("---")
                st.subheader("Simulation Metrics")

                c1, c2, c3, c4, c5 = st.columns(5)

                baseline_delta = (
                    simulation["total_reward"]
                    - baseline["total_reward"]
                )

                with c1:
                    st.metric(
                        "Total Reward",
                        f"€{simulation['total_reward']:.2f}"
                    )

                with c2:
                    st.metric(
                        "Final SOC",
                        f"{simulation['final_soc']:.1f}%"
                    )

                with c3:
                    st.metric(
                        "Charge",
                        simulation["charge"]
                    )

                with c4:
                    st.metric(
                        "Discharge",
                        simulation["discharge"]
                    )

                with c5:
                    st.metric(
                        "vs Price-Based",
                        f"€{baseline_delta:.2f}",
                        delta=f"€{baseline_delta:.2f}",
                    )

                first_row = results.iloc[0]

                st.markdown("---")

                d1, d2, d3 = st.columns(3)

                with d1:
                    st.metric(
                        "Current Price",
                        f"€{first_row['Price']:.2f}/MWh"
                    )

                with d2:
                    st.metric(
                        "Current SOC",
                        f"{first_row['SOC']:.1f}%"
                    )

                with d3:

                    if first_row["Action"] == "CHARGE":
                        st.success("🟢 Current Action: CHARGE")
                    elif first_row["Action"] == "DISCHARGE":
                        st.error("🔴 Current Action: DISCHARGE")
                    else:
                        st.info("⚪ Current Action: IDLE")

                st.markdown("---")

                st.plotly_chart(
                    make_price_action_figure(results),
                    use_container_width=True,
                )

                left, right = st.columns(2)

                with left:
                    st.plotly_chart(
                        make_soc_figure(results),
                        use_container_width=True,
                    )

                with right:

                    cumulative_reward = (
                        results["Reward"].cumsum()
                    )

                    cumulative_fig = go.Figure(
                        go.Scatter(
                            x=results["Hour"],
                            y=cumulative_reward,
                            mode="lines+markers",
                            fill="tozeroy",
                            hovertemplate=(
                                "Hour %{x}<br>"
                                "Cumulative Reward: €%{y:.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    cumulative_fig.update_layout(
                        title="Cumulative Reward",
                        xaxis_title="Hour",
                        yaxis_title="Cumulative Reward (€)",
                        height=430,
                        margin=dict(l=20, r=20, t=60, b=20),
                    )

                    st.plotly_chart(
                        cumulative_fig,
                        use_container_width=True,
                    )

                st.markdown("---")

                a1, a2, a3 = st.columns(3)

                with a1:
                    st.success(
                        f"🟢 Charge: {simulation['charge']}"
                    )

                with a2:
                    st.error(
                        f"🔴 Discharge: {simulation['discharge']}"
                    )

                with a3:
                    st.info(
                        f"⚪ Idle: {simulation['idle']}"
                    )

                st.subheader(
                    "24-Hour Simulation Details"
                )

                display_results = results[
                    [
                        "Hour",
                        "Price",
                        "Action",
                        "Reward",
                        "SOC",
                    ]
                ].copy()

                display_results["Price"] = (
                    display_results["Price"].round(2)
                )
                display_results["Reward"] = (
                    display_results["Reward"].round(3)
                )
                display_results["SOC"] = (
                    display_results["SOC"].round(1)
                )

                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇ Download Simulation CSV",
                    data=results.to_csv(index=False),
                    file_name=(
                        f"{selected_model.lower().replace(' ', '_')}"
                        f"_day_{test_day}_simulation.csv"
                    ),
                    mime="text/csv",
                )

            except Exception as error:

                st.error(
                    f"Simulation failed: {error!r}"
                )

        else:

            st.info(
                "Choose the model and test settings in the sidebar, "
                "then run the simulation."
            )

    # --------------------------------------------------------
    # COMPARE MODE DIRECTLY INSIDE LIVE TAB
    # --------------------------------------------------------

    else:

        st.subheader(
            "Dueling DQN vs Double DQN"
        )

        st.info(
            "Comparison mode runs both trained models on the same "
            "24-hour test period and the same price dataset."
        )

        compare_button = st.button(
            "▶ Compare Both Models",
            type="primary",
            use_container_width=True,
        )

        if compare_button:

            try:

                with st.spinner(
                    "Running Dueling DQN and Double DQN..."
                ):

                    dueling_results = run_model(
                        load_model("Dueling DQN"),
                        daily_prices,
                        battery_capacity,
                    )

                    double_results = run_model(
                        load_model("Double DQN"),
                        daily_prices,
                        battery_capacity,
                    )

                st.markdown("---")
                st.subheader("Model Comparison")

                comparison_display = pd.DataFrame(
                    [
                        {
                            "Metric": "Total Reward",
                            "Dueling DQN":
                                f"€{dueling_results['total_reward']:.2f}",
                            "Double DQN":
                                f"€{double_results['total_reward']:.2f}",
                        },
                        {
                            "Metric": "Final SOC",
                            "Dueling DQN":
                                f"{dueling_results['final_soc']:.1f}%",
                            "Double DQN":
                                f"{double_results['final_soc']:.1f}%",
                        },
                        {
                            "Metric": "Charge Actions",
                            "Dueling DQN":
                                str(dueling_results["charge"]),
                            "Double DQN":
                                str(double_results["charge"]),
                        },
                        {
                            "Metric": "Discharge Actions",
                            "Dueling DQN":
                                str(dueling_results["discharge"]),
                            "Double DQN":
                                str(double_results["discharge"]),
                        },
                        {
                            "Metric": "Idle Actions",
                            "Dueling DQN":
                                str(dueling_results["idle"]),
                            "Double DQN":
                                str(double_results["idle"]),
                        },
                    ]
                )

                st.dataframe(
                    comparison_display,
                    use_container_width=True,
                    hide_index=True,
                )

                left, right = st.columns(2)

                with left:

                    st.markdown("### 🧠 Dueling DQN")

                    st.metric(
                        "Reward",
                        f"€{dueling_results['total_reward']:.2f}"
                    )

                    st.metric(
                        "Final SOC",
                        f"{dueling_results['final_soc']:.1f}%"
                    )

                with right:

                    st.markdown("### 🧠 Double DQN")

                    st.metric(
                        "Reward",
                        f"€{double_results['total_reward']:.2f}"
                    )

                    st.metric(
                        "Final SOC",
                        f"{double_results['final_soc']:.1f}%"
                    )

                reward_dict = {
                    "Dueling DQN":
                        dueling_results["total_reward"],
                    "Double DQN":
                        double_results["total_reward"],
                }

                st.plotly_chart(
                    make_reward_figure(reward_dict),
                    use_container_width=True,
                )

                st.plotly_chart(
                    make_soc_figure(
                        dueling_results["results"],
                        double_results["results"],
                    ),
                    use_container_width=True,
                )

                action_comparison = go.Figure()

                action_comparison.add_trace(
                    go.Scatter(
                        x=dueling_results["results"]["Hour"],
                        y=dueling_results["results"]["Action Number"],
                        mode="lines+markers",
                        name="Dueling DQN",
                    )
                )

                action_comparison.add_trace(
                    go.Scatter(
                        x=double_results["results"]["Hour"],
                        y=double_results["results"]["Action Number"],
                        mode="lines+markers",
                        name="Double DQN",
                    )
                )

                action_comparison.update_layout(
                    title="Action Comparison",
                    xaxis_title="Hour",
                    yaxis_title="Action",
                    height=430,
                    yaxis=dict(
                        tickmode="array",
                        tickvals=[0, 1, 2],
                        ticktext=[
                            "Idle",
                            "Charge",
                            "Discharge",
                        ],
                    ),
                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=20
                    ),
                )

                st.plotly_chart(
                    action_comparison,
                    use_container_width=True,
                )

            except Exception as error:

                st.error(
                    f"Comparison failed: {error!r}"
                )

        else:

            st.info(
                "Click **Compare Both Models** to run the comparison."
            )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with tab_performance:

    st.subheader(
        "Strategy Benchmark Comparison"
    )

    st.write(
        "Compare both RL agents against a price-based rule and "
        "a no-battery baseline on the selected 24-hour period."
    )

    run_benchmark = st.button(
        "▶ Run Benchmark Comparison",
        type="primary",
        use_container_width=True,
    )

    if run_benchmark:

        try:

            with st.spinner(
                "Running benchmark strategies..."
            ):

                dueling = run_model(
                    load_model("Dueling DQN"),
                    daily_prices,
                    battery_capacity,
                )

                double = run_model(
                    load_model("Double DQN"),
                    daily_prices,
                    battery_capacity,
                )

                price_based = run_price_based(
                    daily_prices,
                    battery_capacity,
                )

                no_battery = run_no_battery(
                    daily_prices
                )

            rewards = {
                "Dueling DQN":
                    dueling["total_reward"],
                "Double DQN":
                    double["total_reward"],
                "Price-Based":
                    price_based["total_reward"],
                "No Battery":
                    no_battery["total_reward"],
            }

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Dueling DQN",
                    f"€{dueling['total_reward']:.2f}"
                )

            with c2:
                st.metric(
                    "Double DQN",
                    f"€{double['total_reward']:.2f}"
                )

            with c3:
                st.metric(
                    "Price-Based",
                    f"€{price_based['total_reward']:.2f}"
                )

            with c4:
                st.metric(
                    "No Battery",
                    f"€{no_battery['total_reward']:.2f}"
                )

            st.plotly_chart(
                make_reward_figure(rewards),
                use_container_width=True,
            )

            benchmark_table = pd.DataFrame([
                {
                    "Strategy": "Dueling DQN",
                    "Reward (€)": dueling["total_reward"],
                    "Final SOC (%)": dueling["final_soc"],
                },
                {
                    "Strategy": "Double DQN",
                    "Reward (€)": double["total_reward"],
                    "Final SOC (%)": double["final_soc"],
                },
                {
                    "Strategy": "Price-Based",
                    "Reward (€)": price_based["total_reward"],
                    "Final SOC (%)": price_based["final_soc"],
                },
                {
                    "Strategy": "No Battery",
                    "Reward (€)": no_battery["total_reward"],
                    "Final SOC (%)": 0.0,
                },
            ])

            st.dataframe(
                benchmark_table.round(2),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:

            st.error(
                f"Benchmark failed: {error!r}"
            )

    else:

        st.info(
            "Run the benchmark to compare all strategies."
        )


# ============================================================
# TRAINING ANALYTICS TAB
# ============================================================

with tab_training:

    st.subheader(
        "Training Analytics"
    )

    training_fig, found = make_training_figure(
        [
            "Dueling DQN",
            "Double DQN",
        ]
    )

    if found:
        st.plotly_chart(
            training_fig,
            use_container_width=True,
        )
    else:
        st.warning(
            "Training reward logs were not found."
        )

    st.info(
        "The project currently saves episode rewards but not a "
        "complete per-episode loss history, so the dashboard "
        "does not fabricate a loss curve."
    )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with tab_architecture:

    st.subheader(
        "RL Architecture"
    )

    left, right = st.columns(2)

    with left:

        model = load_model("Dueling DQN")

        st.markdown("### Dueling DQN")

        st.metric(
            "Trainable Parameters",
            f"{count_parameters(model):,}"
        )

        st.code(
            "Q(s,a) = V(s) + A(s,a) - mean(A(s,a))"
        )

        st.write(
            "Separates state-value estimation from "
            "action-advantage estimation."
        )

    with right:

        model = load_model("Double DQN")

        st.markdown("### Double DQN")

        st.metric(
            "Trainable Parameters",
            f"{count_parameters(model):,}"
        )

        st.write(
            "Uses the online network to select the next action "
            "and the target network to evaluate it."
        )

    st.markdown("---")

    st.subheader(
        "Battery Configuration"
    )

    battery_table = pd.DataFrame({
        "Parameter": [
            "Capacity",
            "Initial SOC",
            "Minimum SOC",
            "Maximum SOC",
            "Max Charge Power",
            "Max Discharge Power",
            "Charge Efficiency",
            "Discharge Efficiency",
        ],
        "Value": [
            f"{battery_capacity} kWh",
            "50%",
            "10%",
            "90%",
            "20 kW",
            "20 kW",
            "90%",
            "90%",
        ],
    })

    st.dataframe(
        battery_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Energy Storage Arbitrage | Dueling DQN + Double DQN | "
    "Python • PyTorch • Gymnasium • Streamlit • Plotly"
)
