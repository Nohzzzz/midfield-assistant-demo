# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime
import matplotlib.patches as patches
import os
import io  # si vous voulez aussi gérer Excel en mémoire

# ----------------------------------------
#        FONCTIONS DE GÉNÉRATION
# ----------------------------------------

def generate_enhanced_soccer_data(duration_minutes=15, fps=1):
    np.random.seed(42)
    total_seconds = duration_minutes * 60
    times = np.arange(0, total_seconds, 1 / fps)

    teams_config = {
        "Team_A": {
            "formation": "4-3-3",
            "players": {
                "GK": ["Team_A_1"],
                "DEF": [f"Team_A_{i}" for i in range(2, 6)],
                "MID": [f"Team_A_{i}" for i in range(6, 9)],
                "FWD": [f"Team_A_{i}" for i in range(9, 12)]
            },
            "strategy": "possession",
            "pressure_intensity": 0.7
        },
        "Team_B": {
            "formation": "4-4-2",
            "players": {
                "GK": ["Team_B_1"],
                "DEF": [f"Team_B_{i}" for i in range(2, 6)],
                "MID": [f"Team_B_{i}" for i in range(6, 10)],
                "FWD": [f"Team_B_{i}" for i in range(10, 12)]
            },
            "strategy": "counter_attack",
            "pressure_intensity": 0.5
        }
    }

    data_positions = []
    data_events = []
    data_physical = []
    team_possession = {"Team_A": 0, "Team_B": 0}
    team_shots = {"Team_A": 0, "Team_B": 0}

    match_context = {
        "score": [0, 0],
        "momentum": 0.5,
        "last_ball_contact": None
    }

    def get_role_base_position(team, player_id):
        for role, players in teams_config[team]["players"].items():
            if player_id in players:
                if role == "GK":
                    return (5 if team == "Team_A" else 95, 25)
                elif role == "DEF":
                    return (25 if team == "Team_A" else 75, np.random.uniform(10, 40))
                elif role == "MID":
                    return (50, np.random.uniform(5, 45))
                elif role == "FWD":
                    return (75 if team == "Team_A" else 25, np.random.uniform(15, 35))
        return (50, 25)

    for t in times:
        minute = int(t // 60)
        match_progress = t / total_seconds

        if t > 0 and t % 300 == 0:
            match_context["momentum"] = np.clip(
                match_context["momentum"] + np.random.uniform(-0.2, 0.2), 0, 1
            )

        for team in teams_config:
            strategy = teams_config[team]["strategy"]

            for role, players in teams_config[team]["players"].items():
                for player in players:
                    base_x, base_y = get_role_base_position(team, player)

                    if strategy == "possession":
                        x_variation = np.random.uniform(-8, 8)
                        y_variation = np.random.uniform(-10, 10)
                    else:  # counter_attack
                        x_variation = np.random.uniform(-15, 15)
                        y_variation = np.random.uniform(-15, 15)

                    if team == "Team_A":
                        momentum_effect = 1 + (match_context["momentum"] - 0.5) * 0.3
                    else:
                        momentum_effect = 1 - (match_context["momentum"] - 0.5) * 0.3

                    x = base_x + x_variation * momentum_effect
                    y = base_y + y_variation * momentum_effect

                    if role == "GK":
                        x = max(0, min(x, 100))
                        y = max(10, min(y, 40))

                    fatigue = 0.1 + 0.8 * match_progress * np.random.uniform(0.8, 1.2)
                    speed = np.random.uniform(2, 8) * (1 - fatigue * 0.3)

                    data_positions.append([t, minute, player, team, role, x, y])
                    data_physical.append([t, minute, player, team, speed, fatigue])

        # Génération d'événements aléatoires
        if np.random.rand() < 0.1:
            event_team = "Team_A" if match_context["momentum"] > 0.5 else "Team_B"
            event_type = np.random.choice(
                ["pass", "cross", "shot", "tackle", "foul"],
                p=[0.6, 0.1, 0.15, 0.1, 0.05]
            )

            if event_type in ["shot", "cross"]:
                player = np.random.choice(teams_config[event_team]["players"]["FWD"])
            else:
                player = np.random.choice(
                    teams_config[event_team]["players"]["MID"] +
                    teams_config[event_team]["players"]["DEF"]
                )

            if event_type == "pass":
                x = np.random.uniform(20, 80)
                y = np.random.uniform(5, 45)
                success = np.random.rand() > 0.2
            elif event_type == "shot":
                x = np.random.uniform(
                    80 if event_team == "Team_A" else 20,
                    95 if event_team == "Team_A" else 25
                )
                y = np.random.uniform(15, 35)
                success = np.random.rand() > 0.7
                if success:
                    match_context["score"][0 if event_team == "Team_A" else 1] += 1
                team_shots[event_team] += 1
            else:
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 50)
                success = True

            data_events.append([
                t, minute, player, event_team, event_type,
                x, y, success, match_context["momentum"]
            ])

            if event_type == "pass" and success:
                team_possession[event_team] += 1
                match_context["last_ball_contact"] = event_team

    positions_df = pd.DataFrame(
        data_positions,
        columns=["time", "minute", "player_id", "team", "role", "x", "y"]
    )

    events_df = pd.DataFrame(
        data_events,
        columns=["time", "minute", "player_id", "team", "event_type",
                 "x", "y", "success", "momentum"]
    )

    physical_df = pd.DataFrame(
        data_physical,
        columns=["time", "minute", "player_id", "team", "speed", "fatigue"]
    )

    stats_df = pd.DataFrame({
        "minute": positions_df["minute"].unique(),
        "possession_Team_A": [
            team_possession["Team_A"] / max(
                1, team_possession["Team_A"] + team_possession["Team_B"]
            )
            for _ in positions_df["minute"].unique()
        ],
        "possession_Team_B": [
            team_possession["Team_B"] / max(
                1, team_possession["Team_A"] + team_possession["Team_B"]
            )
            for _ in positions_df["minute"].unique()
        ],
        "shots_Team_A": [team_shots["Team_A"] for _ in positions_df["minute"].unique()],
        "shots_Team_B": [team_shots["Team_B"] for _ in positions_df["minute"].unique()],
        "momentum": [match_context["momentum"] for _ in positions_df["minute"].unique()]
    })

    return positions_df, events_df, physical_df, stats_df, teams_config


# ----------------------------------------
#     FONCTIONS DE PRÉSENTATION & IA
# ----------------------------------------

# Impacts tactiques par zone
TACTIC_IMPACTS = {
    "4-4-2":    {"bas": -0.05, "médian": +0.12, "haut": -0.07},
    "3-5-2":    {"bas": -0.10, "médian": +0.20, "haut": -0.10},
    "4-3-3":    {"bas": +0.05, "médian": -0.10, "haut": +0.05},
    "5-3-2":    {"bas": +0.10, "médian": +0.05, "haut": -0.15},
    "3-4-3":    {"bas": -0.08, "médian": +0.05, "haut": +0.10},
    "4-2-3-1":  {"bas": 0.00,  "médian": +0.15, "haut": -0.05},
    "4-1-4-1":  {"bas": +0.05, "médian": +0.10, "haut": -0.10},
}


def zone_x(x):
    if x < 35:
        return "bas"
    elif x < 65:
        return "médian"
    else:
        return "haut"


def presence_zones(positions, team, minute=None):
    if minute is not None:
        df = positions[(positions["team"] == team) & (positions["minute"] == minute)].copy()
    else:
        df = positions[positions["team"] == team].copy()
    df["zone"] = df["x"].apply(zone_x)
    presence = df["zone"].value_counts(normalize=True)
    return presence.to_dict()


def simulate_tactic_impact(current_presence, tactic_name):
    if tactic_name not in TACTIC_IMPACTS:
        return None
    impact = TACTIC_IMPACTS[tactic_name]
    simulated_presence = {}
    for zone in ["bas", "médian", "haut"]:
        base = current_presence.get(zone, 0)
        change = impact.get(zone, 0)
        simulated_presence[zone] = max(0, base + change)
    total = sum(simulated_presence.values()) or 1.0
    for zone in simulated_presence:
        simulated_presence[zone] /= total
    return simulated_presence


def draw_pitch(ax):
    """
    Dessine un terrain de foot complet en fond (couleur verte sur tout le canevas),
    puis trace les lignes blanches du terrain.
    """
    pitch_color = "#228B22"   # vert foncé des pelouses
    line_color  = "white"

    # 1) Appliquer la même couleur verte à la figure ET à l'axe
    fig = ax.get_figure()
    fig.patch.set_facecolor(pitch_color)
    ax.set_facecolor(pitch_color)

    # 2) Définir les limites du terrain (0–100 en x, 0–50 en y)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # 3) Tracer le contour extérieur
    ax.add_patch(
        patches.Rectangle((0, 0), 100, 50,
                          fill=False, edgecolor=line_color, linewidth=2)
    )
    # Ligne médiane
    ax.plot([50, 50], [0, 50], color=line_color, linewidth=2)

    # Cercle central + point central
    centre_circle = patches.Circle((50, 25), 9.15,
                                   fill=False, edgecolor=line_color, linewidth=2)
    centre_spot  = patches.Circle((50, 25), 0.3, color=line_color)
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)

    # Surface de réparation (côté gauche)
    ax.add_patch(
        patches.Rectangle((0, 13.84), 16.5, 22.32,
                          fill=False, edgecolor=line_color, linewidth=2)
    )
    left_pen_spot = patches.Circle((11, 25), 0.3, color=line_color)
    left_arc      = patches.Arc((11, 25), 18.3, 18.3, angle=0,
                                theta1=310, theta2=50, edgecolor=line_color, linewidth=2)
    ax.add_patch(left_pen_spot)
    ax.add_patch(left_arc)

    # Surface de réparation (côté droit)
    ax.add_patch(
        patches.Rectangle((100 - 16.5, 13.84), 16.5, 22.32,
                          fill=False, edgecolor=line_color, linewidth=2)
    )
    right_pen_spot = patches.Circle((100 - 11, 25), 0.3, color=line_color)
    right_arc      = patches.Arc((100 - 11, 25), 18.3, 18.3, angle=0,
                                 theta1=130, theta2=230, edgecolor=line_color, linewidth=2)
    ax.add_patch(right_pen_spot)
    ax.add_patch(right_arc)

    # Surface de but (gauche)
    ax.add_patch(
        patches.Rectangle((0, 20.16), 5.5, 9.68,
                          fill=False, edgecolor=line_color, linewidth=2)
    )
    # Surface de but (droite)
    ax.add_patch(
        patches.Rectangle((100 - 5.5, 20.16), 5.5, 9.68,
                          fill=False, edgecolor=line_color, linewidth=2)
    )

    # Cages (gauche)
    ax.add_patch(
        patches.Rectangle((-0.5, 21.33), 0.5, 7.33,
                          fill=False, edgecolor=line_color, linewidth=2)
    )
    # Cages (droite)
    ax.add_patch(
        patches.Rectangle((100, 21.33), 0.5, 7.33,
                          fill=False, edgecolor=line_color, linewidth=2)
    )

    # 4) Enlever graduations/axes
    ax.axis('off')



def plot_minute_players_with_alerts(minute, positions_df, alert_recommendations=None):
    """
    Affiche la position des joueurs à la minute donnée, sur un vrai terrain de foot.
    Les joueurs à remplacer (alertes) sont entourés en rouge.
    """
    # Filtrer la minute
    df_minute = positions_df[positions_df['minute'] == minute]
    if df_minute.empty:
        st.warning(f"Aucune donnée pour la minute {minute}")
        return

    df_unique = df_minute.groupby('player_id').first().reset_index()

    # Repérer les joueurs à remplacer
    alert_players = set()
    if isinstance(alert_recommendations, pd.DataFrame):
        recos = alert_recommendations[alert_recommendations['minute'] == minute]['recommendation'].tolist()
    else:
        recos = []
    for rec in recos:
        if "Remplacement" in rec:
            parts = rec.split(":")[1] if ":" in rec else ""
            for pid in parts.split(","):
                p = pid.strip()
                if p:
                    alert_players.add(p)

    # Créer la figure + dessiner le terrain
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_pitch(ax)

    # Tracer les joueurs
    sns.scatterplot(
        data=df_unique,
        x='x', y='y',
        hue='team', style='role',
        s=140, edgecolor='black', linewidth=0.6,
        ax=ax
    )

    # Annoter chaque point par l'ID du joueur
    for _, row in df_unique.iterrows():
        ax.text(
            row['x'] + 0.8, row['y'] + 0.8,
            row['player_id'], color='white', fontsize=8, weight='bold'
        )

    # Marquer les joueurs alertés en cercle rouge
    for _, row in df_unique.iterrows():
        pid = row['player_id']
        if pid in alert_players:
            ax.scatter(
                row['x'], row['y'],
                s=300, facecolors='none', edgecolors='red', linewidths=2
            )

    ax.set_title(f"📍 Positions des joueurs – Minute {minute}", fontsize=18, color='white')
    plt.tight_layout()
    st.pyplot(fig)
# ----------------------------------------
#   FONCTIONS DE VISUALISATION DE LA PRÉSENCE
# ----------------------------------------

def plot_presence_comparison(current_presence, simulated_presence, tactic_name, team, minute):
    """
    Affiche un histogramme comparant la répartition par zone actuelle
    vs. simulée (bar chart).
    """
    zones = ["bas", "médian", "haut"]
    current_vals = [current_presence.get(z, 0) for z in zones]
    sim_vals     = [simulated_presence.get(z, 0) for z in zones]

    x = np.arange(len(zones))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x,       current_vals, width, label="Présence actuelle", color="#1f77b4")
    bars2 = ax.bar(x + width, sim_vals,  width, label=f"Simulation {tactic_name}", color="#ff7f0e")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(zones, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(f"Comparaison présences – {team} (minute {minute})", fontsize=14)
    ax.legend()

    # Ajouter les valeurs au-dessus des barres
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h*100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10
            )

    plt.tight_layout()
    st.pyplot(fig)


def plot_presence_fieldmap_comparison(current_presence, simulated_presence, tactic_name, team, minute):
    """
    Affiche deux sous-graphes (présence actuelle vs simulée) sur fond de vrai terrain.
    """
    zones = ["bas", "médian", "haut"]
    zone_y = {"bas": 15, "médian": 25, "haut": 35}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    titles = ["Présence actuelle", f"Simulation {tactic_name}"]

    for ax, presence_data, title in zip(axes, [current_presence, simulated_presence], titles):
        # 1. Dessiner le terrain
        draw_pitch(ax)

        # 2. Tracer une bulle par zone (taille proportionnelle à la présence)
        for zone in zones:
            presence = presence_data.get(zone, 0)
            size = 3000 * presence  # mise à l’échelle arbitraire
            color = "dodgerblue" if title == "Présence actuelle" else "orange"
            ax.scatter(
                50, zone_y[zone],
                s=size, color=color, edgecolor="white", alpha=0.7
            )
            ax.text(
                50, zone_y[zone] + 2,
                f"{zone} : {presence:.0%}",
                ha="center", color="white", fontsize=11, fontweight="bold"
            )

        ax.set_title(title, fontsize=14, color="white")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.axis("off")

    fig.suptitle(
        f"Répartition des joueurs – {team} (Minute {minute})",
        fontsize=16, color="white"
    )
    plt.tight_layout()
    st.pyplot(fig)


def display_fatigue_table(physical_minute):
    """
    Affiche un tableau HTML avec le niveau de fatigue moyen de chaque joueur.
    """
    avg_fatigue = physical_minute.groupby('player_id')['fatigue'].mean().reset_index()
    avg_fatigue.columns = ["Joueur", "Fatigue Moyenne"]
    avg_fatigue["Niveau de Fatigue"] = avg_fatigue["Fatigue Moyenne"].apply(
        lambda f: "Bas" if f < 0.3 else ("Modéré" if f < 0.6 else "Élevé")
    )
    st.dataframe(avg_fatigue.sort_values(by="Fatigue Moyenne", ascending=False), use_container_width=True)


def extract_features_minute(positions_df, physical_df, events_df, stats_df, minute):
    pos_minute = positions_df[positions_df['minute'] == minute]
    phys_minute = physical_df[physical_df['minute'] == minute]
    events_minute = events_df[events_df['minute'] == minute]
    stats_minute = stats_df[stats_df['minute'] == minute]

    momentum = stats_minute['momentum'].values[0] if not stats_minute.empty else 0.5
    shots = len(events_minute[events_minute['event_type'] == 'shot'])
    fouls = len(events_minute[events_minute['event_type'] == 'foul'])
    fatigue_mean = phys_minute['fatigue'].mean() if not phys_minute.empty else 0.0

    return [momentum, shots, fouls, fatigue_mean]


def predict_score_evolution(features):
    momentum, shots, fouls, fatigue = features
    prob = 0.2 + 0.5 * momentum + 0.1 * min(shots, 5) - 0.1 * min(fouls, 3) - 0.2 * fatigue
    return max(0, min(1, prob))


def run_ia_score_prediction(positions_df, physical_df, events_df, stats_df):
    unique_minutes = sorted(positions_df['minute'].unique())
    recommendations = []

    for minute in unique_minutes:
        features = extract_features_minute(positions_df, physical_df, events_df, stats_df, minute)
        score_prob = predict_score_evolution(features)

        if score_prob > 0.6:
            reco = f"Minute {minute} : Forte probabilité de marquer ({int(score_prob*100)}%)"
        elif score_prob < 0.3:
            reco = f"Minute {minute} : Faible probabilité de marquer ({int(score_prob*100)}%)"
        else:
            reco = f"Minute {minute} : Probabilité moyenne ({int(score_prob*100)}%)"

        recommendations.append({
            "minute": minute,
            "score_prob": score_prob,
            "recommendation": reco
        })

    return pd.DataFrame(recommendations)


def plot_ia_score_predictions_filtered(df_ia_recommendations, threshold, selected_minute):
    """
    Affiche l’évolution de la probabilité IA de marquer minute par minute.
    - Colorie en rouge les points > threshold, en bleu les autres.
    - Trace une ligne horizontale à y = threshold.
    - Affiche la probabilité IA pour la minute sélectionnée sous forme de metric,
      ainsi que le texte de recommandation associé à cette minute (s’il existe).
    """
    # 1. On repère la probabilité et le texte de recommandation pour la minute sélectionnée
    prob_selected = None
    rec_selected = None
    mask_minute = df_ia_recommendations['minute'] == selected_minute
    if mask_minute.any():
        # On récupère la probabilité
        prob_selected = float(df_ia_recommendations.loc[mask_minute, 'score_prob'].iloc[0])
        # On récupère la recommandation textuelle (colonne 'recommendation')
        rec_selected = df_ia_recommendations.loc[mask_minute, 'recommendation'].iloc[0]

    # 2. Création du graphique
    fig, ax = plt.subplots(figsize=(10, 4))

    # a) Ligne de seuil horizontale
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1)

    # b) Tracer chaque point, rouge si > threshold, bleu sinon
    for _, row in df_ia_recommendations.iterrows():
        minute = row['minute']
        prob   = row['score_prob']
        color  = 'r' if prob > threshold else 'b'
        ax.scatter(minute, prob, color=color, s=40)

    # c) Relier tous les points par une ligne noire pour la tendance
    ax.plot(
        df_ia_recommendations['minute'],
        df_ia_recommendations['score_prob'],
        color='black', linestyle='-', linewidth=1, alpha=0.6
    )

    ax.set_ylim(0, 1.0)
    ax.set_xlim(
        df_ia_recommendations['minute'].min() - 1,
        df_ia_recommendations['minute'].max() + 1
    )
    ax.set_xlabel("Minute", fontsize=12)
    ax.set_ylabel("Probabilité IA de marquer", fontsize=12)
    ax.set_title("📈 Évolution IA – Probabilité de marquer", fontsize=14)

    # d) Annoter uniquement les points rouges (au-dessus du seuil) avec leur pourcentage
    for _, row in df_ia_recommendations.iterrows():
        if row['score_prob'] > threshold:
            ax.annotate(
                f"{row['score_prob']*100:.0f}%",
                (row['minute'], row['score_prob']),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=8,
                color='red'
            )

    plt.tight_layout()
    st.pyplot(fig)

    # 3. Afficher le metric pour la minute sélectionnée
    if prob_selected is not None:
        st.metric(
            label=f"🔎 Probabilité IA (minute {selected_minute})",
            value=f"{prob_selected*100:.1f}%",
            delta=None
        )
    else:
        st.info(f"Aucune donnée IA pour la minute {selected_minute}.")

    # 4. Afficher le commentaire (recommandation) associé à cette minute (s’il existe)
    if rec_selected is not None:
        st.write("**💬 Commentaire IA pour cette minute :**")
        st.info(rec_selected)
    else:
        # Si jamais la minute n'existe pas ou n'a pas de recommandation, on ne fait rien.
        pass


def detect_opponent_weaknesses(events, physical, positions, team="Team_B", minute=None):
    weaknesses = []
    if minute is not None:
        events = events[events['minute'] == minute]
        physical = physical[physical['minute'] == minute]
        positions = positions[positions['minute'] == minute]

    fouls = events[(events["team"] == team) & (events["event_type"] == "foul")]
    fouls_count = fouls["player_id"].value_counts()
    for player, count in fouls_count.items():
        if count >= 3:
            weaknesses.append(f"🚩 {player} (adverse) fait trop de fautes ({count})")

    recent_physical = physical[physical["team"] == team]
    fatigue_avg = recent_physical.groupby("player_id")["fatigue"].mean()
    speed_avg = recent_physical.groupby("player_id")["speed"].mean()
    struggling = (fatigue_avg > 0.7) & (speed_avg < 4.0)
    for player in struggling[struggling].index:
        weaknesses.append(f"⚠️ {player} (adverse) montre fatigue")

    defensive_third = positions[(positions["team"] == team) & (positions["x"] < 35)]
    zone_pressure = defensive_third["player_id"].value_counts()
    for player, count in zone_pressure.items():
        if count > 10:
            weaknesses.append(f"📌 Zone de {player} (adverse) très sollicitée")

    if not weaknesses:
        weaknesses.append("✅ Aucun point faible détecté")
    return weaknesses


def generate_tactical_recommendations(
    positions,
    physical,
    events,
    stats,
    total_subs_done=0,
    sub_windows_used=0,
    replaced_players=None,
    is_halftime=False
):
    if replaced_players is None:
        replaced_players = []

    recommendations = []
    unique_minutes = sorted(positions['minute'].unique())

    MAX_TOTAL_REPLACEMENTS = 5
    MAX_SUB_WINDOWS = 3
    MAX_SUBS_PER_WINDOW = 3
    FATIGUE_THRESHOLD = 0.7
    MIDDLE_X_MIN = 35
    MIDDLE_X_MAX = 65

    def score_recommendation(text):
        if "Remplacement" in text:
            return 100
        if "Faiblesse advers" in text:
            return 90
        if "Seulement" in text and "milieu" in text:
            return 80
        if "Trop de fautes" in text:
            return 70
        if "manque de tirs" in text:
            return 65
        if "momentum" in text:
            return 60
        return 50

    for current_minute in unique_minutes:
        positions_minute = positions[positions['minute'] == current_minute]
        physical_minute = physical[physical['minute'] == current_minute]
        events_minute = events[events['minute'] == current_minute]
        stats_minute = stats[stats['minute'] == current_minute]

        raw_recs = []

        # VERIFIER SI ON PEUT REMPLACER
        can_replace = (
            total_subs_done < MAX_TOTAL_REPLACEMENTS and
            (sub_windows_used < MAX_SUB_WINDOWS or is_halftime)
        )
        if can_replace:
            critically_fatigued = physical_minute[
                (physical_minute['fatigue'] > FATIGUE_THRESHOLD) &
                (~physical_minute['player_id'].isin(replaced_players))
            ]
            fatigue_sorted = critically_fatigued.sort_values(by='fatigue', ascending=False)
            subs_remaining = MAX_TOTAL_REPLACEMENTS - total_subs_done
            subs_this_window = min(subs_remaining, MAX_SUBS_PER_WINDOW)
            players_to_replace = fatigue_sorted.head(subs_this_window)['player_id'].tolist()
            if players_to_replace:
                raw_recs.append("🔁 Remplacement(s) : " + ", ".join(players_to_replace))
        else:
            if total_subs_done >= MAX_TOTAL_REPLACEMENTS:
                raw_recs.append("⛔️ Plus de remplacements (5/5)")
            elif sub_windows_used >= MAX_SUB_WINDOWS and not is_halftime:
                raw_recs.append("⛔️ Toutes fenêtres utilisées")

        # VERIFIER NOMBRE DE JOUEURS AU MILIEU
        middle_zone_players = positions_minute[
            (positions_minute['x'] > MIDDLE_X_MIN) & (positions_minute['x'] < MIDDLE_X_MAX)
        ]
        unique_mid_players = middle_zone_players['player_id'].nunique()
        if unique_mid_players < 6:
            raw_recs.append(f"⚠️ Seulement {unique_mid_players} joueurs au milieu")

        fouls = events_minute[events_minute['event_type'] == 'foul']
        failed_passes = events_minute[
            (events_minute['event_type'] == 'pass') & (~events_minute['success'])
        ]
        shots = events_minute[events_minute['event_type'] == 'shot']

        if len(fouls) > 2:
            raw_recs.append("⚠️ Trop de fautes")
        if len(failed_passes) > 3:
            raw_recs.append("⚠️ Trop de pertes de balle")
        if len(shots) < 1:
            raw_recs.append("🎯 Manque de tirs")

        if not stats_minute.empty:
            momentum = stats_minute['momentum'].values[0]
            if momentum < 0.4:
                raw_recs.append("⬆️ Booster défensif")
            elif momentum > 0.6:
                raw_recs.append("⬆️ Profiter domination")

        weaknesses = detect_opponent_weaknesses(events, physical, positions, team="Team_B", minute=current_minute)
        for w in weaknesses:
            raw_recs.append("🎯 Faiblesse adverse : " + w)

        if not raw_recs:
            raw_recs = ["✅ Équipe équilibrée"]

        prioritized = []
        for r in raw_recs:
            rec_type = "IA" if ("⚠️" in r or "🎯" in r or "Remplacement" in r) else "Analyse"
            score = score_recommendation(r)
            prioritized.append({
                "minute": current_minute,
                "type": rec_type,
                "recommendation": r,
                "priority": score
            })

        prioritized = sorted(prioritized, key=lambda x: x["priority"], reverse=True)[:4]
        if prioritized:
            prioritized[0]["recommendation"] = "🔥 PRIORITÉ : " + prioritized[0]["recommendation"]

        recommendations.extend(prioritized)

    return pd.DataFrame(recommendations)


def suggest_pressing_adaptations_df(positions, events, stats, team="Team_A", opponent="Team_B", current_minute=45):
    recs = []

    pos_minute = positions[
        (positions["minute"] == current_minute) & (positions["team"] == opponent)
    ]
    if pos_minute.empty:
        return pd.DataFrame([{
            "minute": current_minute,
            "type": "Pressing",
            "recommendation": "Aucune donnée position adversaire"
        }])

    avg_x = pos_minute["x"].mean()
    bloc_type = "bas" if avg_x < 35 else "médian" if avg_x < 65 else "haut"
    recs.append(f"📊 Bloc adverse : {bloc_type.upper()} (x moyen={avg_x:.1f})")

    if 'x' not in events.columns:
        return pd.DataFrame([{
            "minute": current_minute,
            "type": "Pressing",
            "recommendation": rec
        } for rec in recs])

    recent_events = events[
        (events["minute"] >= current_minute - 5) &
        (events["minute"] <= current_minute) &
        (events["team"] == team)
    ]
    high_press = recent_events[
        recent_events["event_type"].isin(["tackle", "duel"]) &
        (recent_events["x"] > 65)
    ]
    lost_duels = recent_events[
        (recent_events["event_type"] == "duel") &
        (~recent_events["success"]) &
        (recent_events["x"] > 65)
    ]

    ps = len(high_press)
    pf = len(lost_duels)

    if ps < pf:
        recs.append(f"⚠️ Pressing inefficace : {ps} récup vs {pf} pertes")
        recs.append("🔄 Adapter pressing")
    elif ps > 2 * pf:
        recs.append(f"✅ Pressing efficace : {ps} récup")
        recs.append("⬆️ Maintenir pressing haut")
    else:
        recs.append(f"🔍 Pressing équilibré : {ps} récup, {pf} pertes")
        recs.append("🎯 Ajustements individuels")

    prev_events = events[
        (events["minute"] >= current_minute - 10) &
        (events["minute"] < current_minute - 5) &
        (events["team"] == team)
    ]
    prev_high = prev_events[
        prev_events["event_type"].isin(["tackle", "duel"]) &
        (prev_events["x"] > 65)
    ]
    prev_lost = prev_events[
        (prev_events["event_type"] == "duel") &
        (~prev_events["success"]) &
        (prev_events["x"] > 65)
    ]

    prev_ps = len(prev_high)
    prev_pf = len(prev_lost)
    if prev_ps or prev_pf:
        delta_s = ps - prev_ps
        delta_f = pf - prev_pf
        if delta_s > 1 and delta_f <= 0:
            recs.append("📈 Pressing s'améliore")
        elif delta_s < 0 and delta_f > 0:
            recs.append("📉 Pressing en déclin")
        else:
            recs.append("↔️ Pressing stable")

    recent_stats = stats[stats["minute"] == current_minute]
    if not recent_stats.empty:
        momentum = recent_stats["momentum"].values[0]
        if momentum < 0.4:
            recs.append("⬇️ Perte de momentum - baisser pressing")
        elif momentum > 0.6:
            recs.append("⬆️ Domination - augmenter pressing")

    return pd.DataFrame([{
        "minute": current_minute,
        "type": "Pressing",
        "recommendation": r
    } for r in recs])


def calibrate_team_style(positions_df, team_name):
    team_positions = positions_df[positions_df['team'] == team_name].copy()
    team_positions['zone'] = team_positions['x'].apply(zone_x)
    zone_counts = team_positions['zone'].value_counts(normalize=True).to_dict()
    preferred_zone = max(zone_counts, key=zone_counts.get)

    tactic_scores = {}
    for tactic, impact in TACTIC_IMPACTS.items():
        score = sum(zone_counts.get(z, 0) * impact.get(z, 0) for z in ['bas', 'médian', 'haut'])
        tactic_scores[tactic] = score
    preferred_tactic = max(tactic_scores, key=tactic_scores.get)

    profile = {
        "zone_preference": preferred_zone,
        "preferred_tactic": preferred_tactic,
        "zone_distribution": zone_counts,
        "tactic_scores": tactic_scores
    }
    return profile


def export_match_data_structured(
    positions_df,
    events_df,
    physical_df,
    stats_df,
    match_id,
    team_name,
    export_base_folder="exports"
):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_folder = os.path.join(export_base_folder, f"{date_str}_match_{match_id}_{team_name}")
    os.makedirs(match_folder, exist_ok=True)

    for df in [positions_df, events_df, physical_df, stats_df]:
        df['match_id'] = match_id
        df['team_name'] = team_name
        df['export_date'] = date_str

    positions_df.to_csv(os.path.join(match_folder, "positions.csv"), index=False)
    events_df.to_csv(os.path.join(match_folder, "events.csv"), index=False)
    physical_df.to_csv(os.path.join(match_folder, "physical.csv"), index=False)
    stats_df.to_csv(os.path.join(match_folder, "stats.csv"), index=False)

    with open(os.path.join(match_folder, "README.txt"), "w") as f:
        f.write(f"Match Data Export\n")
        f.write(f"Date export: {date_str}\n")
        f.write(f"Match ID: {match_id}\n")
        f.write(f"Team: {team_name}\n")
        f.write("Files:\n")
        f.write(" - positions.csv : Positions des joueurs\n")
        f.write(" - events.csv : Événements (passes, tirs, fautes)\n")
        f.write(" - physical.csv : Données physiques (fatigue, vitesse)\n")
        f.write(" - stats.csv : Statistiques globales\n")

    return match_folder


# ----------------------------------------
# ----------------------------------------
#            INTERFACE STREAMLIT
# ----------------------------------------

st.set_page_config(layout="wide", page_title="Assistant Tactique Foot IA")

@st.cache_data
def load_data():
    return generate_enhanced_soccer_data()

positions_df, events_df, physical_df, stats_df, teams_config = load_data()
teams = list(teams_config.keys())

# -------------------------------
# Sidebar – Paramètres généraux
# -------------------------------
st.sidebar.title("⚙️ Paramètres généraux")

selected_minute = st.sidebar.slider(
    "⏱️ Sélectionnez la minute (0-∞)",
    int(positions_df['minute'].min()),
    int(positions_df['minute'].max()),
    5
)

selected_team = st.sidebar.selectbox(
    "🏳️ Sélectionnez l'équipe",
    teams
)

selected_tactic = st.sidebar.selectbox(
    "📐 Tactique à simuler",
    list(TACTIC_IMPACTS.keys())
)

view_type = st.sidebar.radio(
    "🖼️ Vue tactique",
    ["Terrain", "Histogramme"]
)

# ← N’oublie pas d’ajouter ce slider juste ici !
ia_threshold = st.sidebar.slider(
    "🔍 Seuil ‘Haute probabilité’ (IA)",
    min_value=0.0, max_value=1.0, value=0.6, step=0.01
)

# ---------------
#  Page principale
# ---------------
st.title("⚽ Assistant Tactique IA - Football")
st.markdown("""
Cette application est un **assistant tactique** pour analyser et accompagner un match de football :  
- 🧠 Génère des recommandations tactiques basées sur la position, la fatigue, et l’état du jeu.  
- 💪 Affiche la **fatigue** de chaque joueur minute par minute.  
- 📍 Affiche les **positions des joueurs** sur un vrai terrain, avec alertes visuelles pour les remplacements.  
- 🧪 Permet de **simuler l’impact d’une tactique** en temps réel.  
- 🤖 Propose une **prédiction IA** de la probabilité de marquer en fonction des événements.  
- 📊 Synthétise un **profil tactique** de l’équipe (zone préférée, répartition, tactique recommandée).  
- 💾 Offre la possibilité d’**exporter** toutes les données au format CSV.  
""")

# ----------------------------------------------------------------------------------------------------
# On regroupe les blocs de sortie en onglets ("tabs") pour alléger la navigation et structurer la page.
# ----------------------------------------------------------------------------------------------------
tabs = st.tabs([
    "📋 Recos & Fatigue",
    "📍 Positions sur terrain",
    "🧪 Simulation Tactique",
    "🤖 Prédiction IA",
    "📊 Profil & Export"
])

# =================================
# Onglet 1 : Recommandations & Fatigue
# =================================
with tabs[0]:
    st.subheader("📋 Recommandations Tactiques & Pressing")
    st.markdown(
        "⇢ **Explications :**\n"
        "- Ce tableau liste les recommandations tactiques minute par minute (remplacements, ajustements de position, faiblesses adverses…)\n"
        "- En dessous, l’état de **pressing** est analysé pour l’équipe sélectionnée.\n"
        "- Utilisez le slider à gauche pour changer la minute et l’équipe."
    )
    # Générer et afficher les recommandations tactiques + pressing
    tactical_df = generate_tactical_recommendations(
        positions_df, physical_df, events_df, stats_df
    )
    pressing_df = suggest_pressing_adaptations_df(
        positions_df, events_df, stats_df, current_minute=selected_minute
    )
    all_recs_df = pd.concat([tactical_df, pressing_df], ignore_index=True)
    minute_recs = all_recs_df.loc[
        all_recs_df['minute'] == selected_minute,
        ['type', 'recommendation']
    ]

    if minute_recs.empty:
        st.info(f"Aucune recommandation pour la minute {selected_minute}.")
    else:
        # On utilise deux colonnes : à gauche le tableau, à droite un texte explicatif si besoin
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(minute_recs, use_container_width=True)
        with col2:
            st.info(
                "✅ **Légende des types de recommandations :**\n"
                "- 🔥 PRIORITÉ : la recommandation la plus urgente.\n"
                "- IA : suggérée directement par l’IA (faiblesse, probabilité élevée…)\n"
                "- Analyse : basée sur des règles métier (fatigue, positionnement…)"
            )

    st.markdown("---")
    st.subheader("💪 Niveau de Fatigue des Joueurs")
    st.markdown(
        "⇢ **Explications :**\n"
        "- Ce tableau montre la **fatigue moyenne** de chaque joueur à la minute sélectionnée.\n"
        "- Permet d’anticiper les remplacements ou changements de rythme.\n"
        "- Les joueurs en **“Élevé”** sont candidats au remplacement."
    )
    physical_min = physical_df[physical_df['minute'] == selected_minute]
    if physical_min.empty:
        st.info(f"Aucune donnée physique pour la minute {selected_minute}.")
    else:
        display_fatigue_table(physical_min)

# =================================
# Onglet 2 : Positions sur terrain
# =================================
with tabs[1]:
    st.subheader("📍 Positions des Joueurs – Minute {}".format(selected_minute))
    st.markdown(
        "⇢ **Explications :**\n"
        "- Visualisez en temps réel la position de chaque joueur sur un **terrain de foot** complet.\n"
        "- Les joueurs à remplacer (recommandation “Remplacement”) sont entourés en rouge.\n"
        "- Utile pour repérer visuellement la densité par zone ou le positionnement adverse."
    )
    plot_minute_players_with_alerts(
        selected_minute,
        positions_df,
        alert_recommendations=tactical_df
    )

# =================================
# Onglet 3 : Simulation Tactique
# =================================
with tabs[2]:
    st.subheader("🧪 Simulation d’Impact Tactique")
    st.markdown(
        "⇢ **Explications :**\n"
        "- Comparez la répartition **actuelle** des joueurs par zone (bas, médian, haut) à une simulation\n"
        "  où l’équipe change de tactique (4-4-2, 3-5-2, etc.).\n"
        "- Choisissez la **tactique** dans la barre latérale et la **vue** (Terrain ou Histogramme).\n"
        "- En mode “Terrain”, vous verrez des bulles de taille proportionnelle à la présence dans chaque zone."
    )
    # Récupérer les données de présence actuelle + simulation
    current_presence = presence_zones(
        positions_df, selected_team, selected_minute
    )
    simulated_presence = simulate_tactic_impact(
        current_presence, selected_tactic
    )

    if simulated_presence is None:
        st.error(f"Tactique '{selected_tactic}' inconnue.")
    else:
        if view_type == "Terrain":
            plot_presence_fieldmap_comparison(
                current_presence,
                simulated_presence,
                selected_tactic,
                selected_team,
                selected_minute
            )
        else:
            plot_presence_comparison(
                current_presence,
                simulated_presence,
                selected_tactic,
                selected_team,
                selected_minute
            )

# =================================
# Onglet 4 : Prédiction IA
# =================================
with tabs[3]:
    st.subheader("🤖 Prédiction IA – Chances de Marquer")
    st.markdown(
        "⇢ **Explications :**\n"
        "- L’IA calcule, pour chaque minute du match, la probabilité de marquer\n"
        "  en fonction des événements (vitesse, fatigue, tirs, fautes, momentum).\n"
        "- Le **slider ‘Seuil IA’** dans la barre latérale permet de définir la barre\n"
        "  au-dessus de laquelle on considère la probabilité comme “élevée”.\n"
        "- Les points sont **rouges** si supérieurs à ce seuil, **bleus** sinon.\n"
        "- On affiche également, sous forme de **metric**, la probabilité précise\n"
        "  pour la minute sélectionnée, accompagnée du commentaire IA associé."
    )
    # Calcul de la DataFrame IA
    df_ia = run_ia_score_prediction(
        positions_df, physical_df, events_df, stats_df
    )
    # On appelle la fonction améliorée en passant le seuil et la minute
    plot_ia_score_predictions_filtered(
        df_ia, ia_threshold, selected_minute
    )

# =================================
# Onglet 5 : Profil & Export
# =================================
with tabs[4]:
    st.subheader("📊 Profil Tactique Estimé & Export")
    st.markdown(
        "⇢ **Explications :**\n"
        "- **Profil Tactique** : synthèse des zones préférées et tactiques les plus adaptées.\n"
        "- Ce profil se base sur la **distribution historique** des joueurs sur le terrain.\n"
        "- **Export des données** : choisissez le dossier sur votre machine locale pour enregistrer\n"
        "  les fichiers CSV générés (positions, événements, physique, statistiques)."
    )

    # ----------------------------------------------------
    # 5.1 Affichage du Profil Tactique (zone, répartition, tactique)
    # ----------------------------------------------------
    profile = calibrate_team_style(positions_df, selected_team)
    zone_pref = profile["zone_preference"]
    tactic_rec = profile["preferred_tactic"]
    zone_dist = profile["zone_distribution"]

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown(f"**Zone privilégiée :** {zone_pref.upper()}")
        st.markdown(f"**Tactique recommandée :** {tactic_rec}")
        st.markdown("**Distribution actuelle par zone (en %) :**")
        df_zone = pd.DataFrame({
            "Zone": ["BAS", "MÉDIAN", "HAUT"],
            "Présence (%)": [
                100 * zone_dist.get("bas", 0),
                100 * zone_dist.get("médian", 0),
                100 * zone_dist.get("haut", 0)
            ]
        })
        df_zone["Présence (%)"] = df_zone["Présence (%)"].round(1)
        st.table(df_zone)

    with colB:
        st.info(
            "ℹ️ **Notes pour le staff :**\n"
            "- La zone privilégiée est celle où l’équipe a le plus joué.\n"
            "- La tactique recommandée maximise la couverture des zones clés.\n"
            "- Utilisez ces infos pour ajuster le plan de jeu."
        )

    st.markdown("---")

    # ----------------------------------------------------
    # 5.2 Choix du dossier d’export puis export CSV
    # ----------------------------------------------------
    st.subheader("💾 Export des données du match (CSV)")

    # 1) Champ texte pour que l’utilisateur entre le chemin du dossier où il veut enregistrer
    export_dir = st.text_input(
        "📂 Chemin du dossier d’export (sur votre machine locale)",
        value=""  # vous pouvez proposer un chemin par défaut si vous voulez
    )

    st.markdown(
        "Entrez ici le **chemin absolu** du dossier dans lequel vous souhaitez sauvegarder\n"
        "les fichiers CSV générés. Exemple sous macOS/Linux : `/Users/monNom/Downloads/export_foot/`\n"
        "ou sous Windows : `C:\\Users\\monNom\\Downloads\\export_foot\\`."
    )

    # 2) Bouton pour lancer l’export (ne fonctionne que si export_dir existe)
    if st.button("📝 Lancer l’export CSV"):
        if export_dir.strip() == "":
            st.error("⚠️ Vous devez indiquer un chemin de dossier valide.")
        else:
            # Vérifier que le dossier existe
            if not os.path.isdir(export_dir):
                st.error(f"⚠️ Le dossier n’existe pas : {export_dir}")
            else:
                # Appeler la fonction d’export, en lui passant le dossier choisi
                # export_match_data_structured créera un sous-dossier horodaté à l’intérieur de export_dir
                dossier_export = export_match_data_structured(
                    positions_df,
                    events_df,
                    physical_df,
                    stats_df,
                    match_id="demo_match",
                    team_name=selected_team,
                    export_base_folder=export_dir
                )
                st.success(f"✅ Données CSV exportées dans : {dossier_export}")

