
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
import plotly.graph_objects as go
from matplotlib import patheffects

# -----------------------------
# ⚙️ Paramètres
# -----------------------------
MIDDLE_X_MIN = 35
MIDDLE_X_MAX = 70
ALERT_THRESHOLD_PASSES = 0.4
MINUTES_WINDOW = 2

# -----------------------------
# 📦 Données synthétiques
# -----------------------------
@st.cache_data
def generate_synthetic_data(duration_minutes=15, fps=1):
    np.random.seed(42)
    total_seconds = duration_minutes * 60
    times = np.arange(0, total_seconds, 1/fps)
    players_team_a = [f"Team_A_{i+1}" for i in range(10)]
    players_team_b = [f"Team_B_{i+1}" for i in range(10)]
    data_positions = []
    data_events = []
    for t in times:
        for player in players_team_a:
            x = np.random.uniform(40, 60)
            y = np.random.uniform(10, 40)
            data_positions.append([t, player, "Team_A", x, y])
        for player in players_team_b:
            x = np.random.uniform(35, 65)
            y = np.random.uniform(5, 45)
            data_positions.append([t, player, "Team_B", x, y])
        if np.random.rand() < (1/10):
            passer_team = "Team_A" if np.random.rand() < 0.6 else "Team_B"
            passer_player = np.random.choice(players_team_a if passer_team == "Team_A" else players_team_b)
            pass_x = np.random.uniform(35, 70)
            pass_y = np.random.uniform(0, 50)
            data_events.append([t, passer_player, passer_team, "pass", pass_x, pass_y])
    positions_df = pd.DataFrame(data_positions, columns=["time", "player_id", "team", "x", "y"])
    events_df = pd.DataFrame(data_events, columns=["time", "player_id", "team", "event_type", "x", "y"])
    positions_df['minute'] = positions_df['time'] // 60
    events_df['minute'] = events_df['time'] // 60
    return positions_df, events_df

# -----------------------------
# 📊 Alertes & recommandations
# -----------------------------
def generate_recommendation(minute, pos_window):
    team_a_positions = pos_window[pos_window['team'] == 'Team_A']
    team_b_positions = pos_window[pos_window['team'] == 'Team_B']

    centroid_a_x = team_a_positions['x'].mean()
    centroid_b_x = team_b_positions['x'].mean()

    recommendation = []

    if centroid_a_x < centroid_b_x - 5:
        recommendation.append("Poussez le milieu vers l'avant pour équilibrer le terrain.")

    in_middle = pos_window[(pos_window['x'] >= MIDDLE_X_MIN) & (pos_window['x'] <= MIDDLE_X_MAX)]
    team_a_mid_count = in_middle[in_middle['team'] == "Team_A"]['player_id'].nunique()
    team_b_mid_count = in_middle[in_middle['team'] == "Team_B"]['player_id'].nunique()

    if team_a_mid_count < team_b_mid_count:
        recommendation.append("Augmentez la présence au milieu pour reprendre le contrôle.")

    if not recommendation:
        recommendation.append("Surveillez la possession et ajustez les passes.")

    return recommendation

def compute_alerts(positions, events):
    alerts = []
    possession_dict = {}
    alert_recommendations = {}

    for minute in range(int(positions['minute'].min()), int(positions['minute'].max()) - MINUTES_WINDOW):
        pos_window = positions[(positions['minute'] >= minute) & (positions['minute'] < minute + MINUTES_WINDOW)]
        ev_window = events[(events['minute'] >= minute) & (events['minute'] < minute + MINUTES_WINDOW)]

        in_middle = pos_window[(pos_window['x'] >= MIDDLE_X_MIN) & (pos_window['x'] <= MIDDLE_X_MAX)]
        team_a_count = in_middle[in_middle['team'] == "Team_A"]['player_id'].nunique()
        team_b_count = in_middle[in_middle['team'] == "Team_B"]['player_id'].nunique()

        passes_middle = ev_window[(ev_window['event_type'] == "pass") & 
                                  (ev_window['x'] >= MIDDLE_X_MIN) & (ev_window['x'] <= MIDDLE_X_MAX)]
        team_a_passes = passes_middle[passes_middle['team'] == "Team_A"].shape[0]
        total_passes = passes_middle.shape[0]
        ratio = team_a_passes / total_passes if total_passes > 0 else 0.5
        possession_dict[minute] = ratio

        if team_a_count < team_b_count and ratio < ALERT_THRESHOLD_PASSES:
            alerts.append(minute)
            alert_recommendations[minute] = generate_recommendation(minute, pos_window)

    forced_minute = 9
    if forced_minute in positions['minute'].unique():
        if forced_minute not in alerts:
            alerts.append(forced_minute)
        pos_window_forced = positions[(positions['minute'] >= forced_minute) & (positions['minute'] < forced_minute + MINUTES_WINDOW)]
        alert_recommendations[forced_minute] = generate_recommendation(forced_minute, pos_window_forced)

    return alerts, possession_dict, alert_recommendations

# -----------------------------
# 🎬 Interface Streamlit
# -----------------------------
st.set_page_config(layout="wide", page_title="Midfield Assistant")
st.title("⚽ Midfield Assistant - Analyse Tactique")

positions, events = generate_synthetic_data()
alerts, possession_dict, alert_recommendations = compute_alerts(positions, events)

minute = st.slider("🕒 Choisissez la minute", int(positions['minute'].min()), int(positions['minute'].max()), 0)

st.subheader(f"📍 Position des joueurs à la minute {minute}")
pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
fig, ax = pitch.draw(figsize=(10, 6))

frame = positions[positions['minute'] == minute].sort_values('time').groupby(['player_id', 'team']).tail(1)
team_a = frame[frame['team'] == 'Team_A']
team_b = frame[frame['team'] == 'Team_B']

pitch.scatter(team_a['x'], team_a['y'], ax=ax, s=200, color='blue', label='Team_A')
pitch.scatter(team_b['x'], team_b['y'], ax=ax, s=200, color='red', label='Team_B')

for _, row in frame.iterrows():
    num = row['player_id'].split('_')[-1]
    ax.text(row['x'], row['y'] + 1.3, num, color='white', ha='center', fontsize=14, weight='bold',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])

ax.set_title(f"Minute {minute}", fontsize=16)

if minute in alerts:
    st.warning("⚠️ Alerte : perte de contrôle détectée")
    for rec in alert_recommendations.get(minute, []):
        st.info(f"💡 {rec}")

st.pyplot(fig)

st.subheader("📈 Ratio de passes au milieu (Team_A vs Team_B)")
minutes = sorted(possession_dict.keys())
ratios_team_a = [possession_dict[m] for m in minutes]
ratios_team_b = [1 - possession_dict[m] for m in minutes]

fig_ratio = go.Figure()
fig_ratio.add_trace(go.Scatter(x=minutes, y=ratios_team_a, mode='lines+markers', name='Team A', line=dict(color='blue')))
fig_ratio.add_trace(go.Scatter(x=minutes, y=ratios_team_b, mode='lines+markers', name='Team B', line=dict(color='red')))
fig_ratio.add_hline(y=ALERT_THRESHOLD_PASSES, line_dash="dash", line_color="gray",
                    annotation_text="Seuil alerte", annotation_position="bottom right")
fig_ratio.update_layout(
    xaxis_title="Minute",
    yaxis_title="Ratio passes au milieu",
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig_ratio, use_container_width=True)

st.subheader("🔥 Heatmaps par tranche de 10 minutes")

def plot_heatmap_realistic(team, df, start_min, end_min):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(f"{team}_map", ["#d3d3d3", "blue"] if team == "Team_A" else ["#d3d3d3", "darkred"])

    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(6, 4))

    data = df[(df['minute'] >= start_min) & (df['minute'] < end_min) & (df['team'] == team)]
    sns.kdeplot(data=data, x='x', y='y', ax=ax, fill=True,
                cmap=cmap, alpha=0.6, bw_adjust=0.5, thresh=0.05, levels=100)

    ax.set_title(f"{team} - Minutes {start_min}-{end_min}")
    return fig

max_minute = int(positions['minute'].max())
period_length = 10

for start_min in range(0, max_minute, period_length):
    end_min = start_min + period_length
    st.markdown(f"### ⏱️ Période : {start_min} - {end_min}")
    col1, col2 = st.columns(2)

    with col1:
        fig_a = plot_heatmap_realistic("Team_A", positions, start_min, end_min)
        st.pyplot(fig_a)

    with col2:
        fig_b = plot_heatmap_realistic("Team_B", positions, start_min, end_min)
        st.pyplot(fig_b)
