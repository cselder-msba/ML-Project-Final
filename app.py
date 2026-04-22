import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Player Scouting Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container { padding-top: 2rem; max-width: 1200px; }

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.dash-header h1 {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    color: #f1f5f9;
}
.dash-header .sub {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Stat cards */
.stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
}

/* Player chips */
.player-chip {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px 4px;
    color: #fff;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    host = os.environ["DB_HOST"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    name = os.environ["DB_NAME"]
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:3306/{name}")
    return pd.read_sql("SELECT * FROM player_profiles", engine)

df = load_data()

# ── Stat definitions ─────────────────────────────────────────────────────────
# Using the adjusted stats for fair cross-league comparison
STAT_GROUPS = {
    "Passing": {
        "pass_completion_rate_adj": "Pass Completion %",
        "avg_pass_length_adj": "Avg Pass Length",
        "progressive_pass_rate_adj": "Progressive Pass %",
        "cross_rate_adj": "Cross Rate",
        "switch_rate_adj": "Switch Rate",
        "through_ball_rate_adj": "Through Ball Rate",
        "shot_assist_rate_adj": "Shot Assist Rate",
        "goal_assist_rate_adj": "Goal Assist Rate",
        "pressure_pass_rate_adj": "Pressure Pass %",
    },
    "Shooting": {
        "avg_xg_per_shot_adj": "Avg xG/Shot",
        "shot_on_target_rate_adj": "Shot on Target %",
    },
    "Dribbling & Duels": {
        "dribble_success_rate_adj": "Dribble Success %",
        "duel_win_rate_adj": "Duel Win %",
    },
    "Carrying": {
        "avg_carry_dist_adj": "Avg Carry Distance",
        "final_third_rate_adj": "Final Third %",
    },
    "Volume (per 100)": {
        "pass_per_100_adj": "Passes/100",
        "shot_per_100_adj": "Shots/100",
        "dribble_per_100_adj": "Dribbles/100",
        "duel_per_100_adj": "Duels/100",
        "carry_per_100_adj": "Carries/100",
    },
    "Pressure & Activity": {
        "under_pressure_rate_adj": "Under Pressure %",
    },
}

ALL_STATS = {}
for group in STAT_GROUPS.values():
    ALL_STATS.update(group)

# Preset profiles for quick selection
STAT_PRESETS = {
    # Attacking
    "Winger": [
        "dribble_success_rate_adj", "dribble_per_100_adj", "cross_rate_adj",
        "final_third_rate_adj", "shot_assist_rate_adj", "goal_assist_rate_adj",
        "avg_carry_dist_adj", "progressive_pass_rate_adj",
    ],
    "Striker": [
        "avg_xg_per_shot_adj", "shot_on_target_rate_adj", "shot_per_100_adj",
        "goal_assist_rate_adj", "duel_win_rate_adj", "final_third_rate_adj",
        "dribble_success_rate_adj", "through_ball_rate_adj",
    ],
    # Midfield
    "Deep-Lying Midfielder": [
        "pass_completion_rate_adj", "avg_pass_length_adj", "switch_rate_adj",
        "progressive_pass_rate_adj", "pressure_pass_rate_adj",
        "duel_win_rate_adj", "duel_per_100_adj", "under_pressure_rate_adj",
    ],
    "All-Around Midfielder": [
        "pass_completion_rate_adj", "progressive_pass_rate_adj", "shot_assist_rate_adj",
        "dribble_success_rate_adj", "duel_win_rate_adj", "carry_per_100_adj",
        "final_third_rate_adj", "under_pressure_rate_adj",
    ],
    "Attacking Midfielder": [
        "through_ball_rate_adj", "shot_assist_rate_adj", "goal_assist_rate_adj",
        "avg_xg_per_shot_adj", "dribble_success_rate_adj",
        "progressive_pass_rate_adj", "final_third_rate_adj", "pass_per_100_adj",
    ],
    # Defense
    "Fullback": [
        "cross_rate_adj", "progressive_pass_rate_adj", "avg_carry_dist_adj",
        "duel_win_rate_adj", "pass_completion_rate_adj",
        "under_pressure_rate_adj", "final_third_rate_adj", "carry_per_100_adj",
    ],
    "Center Half": [
        "duel_win_rate_adj", "duel_per_100_adj", "avg_pass_length_adj",
        "pass_completion_rate_adj", "pressure_pass_rate_adj",
        "under_pressure_rate_adj", "switch_rate_adj", "progressive_pass_rate_adj",
    ],
    # Technical
    "Passing": [
        "pass_completion_rate_adj", "avg_pass_length_adj", "progressive_pass_rate_adj",
        "cross_rate_adj", "switch_rate_adj", "through_ball_rate_adj",
        "shot_assist_rate_adj", "pressure_pass_rate_adj",
    ],
    "Shooting": [
        "avg_xg_per_shot_adj", "shot_on_target_rate_adj", "shot_per_100_adj",
        "goal_assist_rate_adj", "shot_assist_rate_adj",
        "dribble_success_rate_adj", "final_third_rate_adj", "through_ball_rate_adj",
    ],
}

PLAYER_COLORS = ["#3b82f6", "#f97316", "#22c55e", "#ec4899"]

def hex_to_rgba(hex_color, alpha=0.08):
    """Convert hex color to rgba format with optional alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <h1>⚽ Player Scouting Dashboard</h1>
    <div class="sub">K-Means Clustering · 4,191 Players · Position-Based Behavioral Profiles</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Position Group")
    position_group = st.selectbox(
        "Filter by position",
        options=["All"] + sorted(df["broad_position"].unique().tolist()),
        label_visibility="collapsed",
    )

    filtered = df if position_group == "All" else df[df["broad_position"] == position_group]

    st.markdown("### Sub Group")
    sub_groups = sorted(filtered["sub_group"].dropna().unique().tolist())
    sub_group = st.selectbox(
        "Filter by sub-group",
        options=["All"] + sub_groups,
        label_visibility="collapsed",
    )
    if sub_group != "All":
        filtered = filtered[filtered["sub_group"] == sub_group]

    st.markdown("### Cluster")
    clusters = sorted(filtered["cluster_full"].dropna().unique().tolist())
    cluster_choice = st.selectbox(
        "Filter by cluster",
        options=["All"] + clusters,
        label_visibility="collapsed",
    )
    if cluster_choice != "All":
        filtered = filtered[filtered["cluster_full"] == cluster_choice]

    st.divider()
    st.markdown("### FM Role")
    fm_roles = sorted(filtered["fm_role"].dropna().unique().tolist())
    fm_role_choice = st.selectbox(
        "Filter by FM role",
        options=["All"] + fm_roles,
        label_visibility="collapsed",
    )
    if fm_role_choice != "All":
        filtered = filtered[filtered["fm_role"] == fm_role_choice]

    st.divider()
    st.markdown("### Competition")
    competitions = sorted(filtered["primary_competition"].dropna().unique().tolist())
    comp_choice = st.selectbox(
        "Filter by competition",
        options=["All"] + competitions,
        label_visibility="collapsed",
    )
    if comp_choice != "All":
        filtered = filtered[filtered["primary_competition"] == comp_choice]

    st.divider()
    st.caption(f"**{len(filtered):,}** players in current filter")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_compare, tab_finder, tab_explore, tab_roster = st.tabs(["Compare Players", "Similar Players", "Stats Explorer", "Player Table"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: COMPARE PLAYERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("#### Select up to 4 players")
    player_names = sorted(filtered["player_name"].dropna().unique().tolist())

    selected_players = st.multiselect(
        "Pick players to compare",
        options=player_names,
        max_selections=4,
        label_visibility="collapsed",
        placeholder="Search for a player...",
    )
    
    # Filter out any previously selected players that no longer match current filters
    selected_players = [p for p in selected_players if p in player_names]

    # Stat selection
    col_preset, col_custom = st.columns([1, 2])
    with col_preset:
        preset = st.selectbox("Stat Preset", options=["Custom"] + list(STAT_PRESETS.keys()))

    if preset != "Custom":
        default_stats = STAT_PRESETS[preset]
    else:
        default_stats = list(STAT_PRESETS["All-Around Midfielder"])

    with col_custom:
        chosen_stats = st.multiselect(
            "Select stats for radar",
            options=list(ALL_STATS.keys()),
            default=default_stats,
            format_func=lambda x: ALL_STATS.get(x, x),
            label_visibility="collapsed",
        )

    if selected_players and chosen_stats:
        player_data = filtered[filtered["player_name"].isin(selected_players)]

        # ── Player info cards ────────────────────────────────────────────
        cols = st.columns(len(selected_players))
        for i, pname in enumerate(selected_players):
            prow = player_data[player_data["player_name"] == pname].iloc[0]
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            with cols[i]:
                st.markdown(f"""
                <div class="stat-card" style="border-top: 3px solid {color};">
                    <div class="value" style="font-size:16px; color:{color};">{pname}</div>
                    <div class="label" style="margin-top:8px;">{prow.get('primary_position', '—')}</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
                        {prow.get('primary_competition', '—')} · Cluster: {prow.get('cluster_full', '—')}
                    </div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                        FM Role: {prow.get('fm_role', '—')} · Fit: {prow.get('role_fit_score', '—')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Normalize stats for radar (percentile within filtered group) ─
        stat_labels = [ALL_STATS[s] for s in chosen_stats]

        fig = go.Figure()

        for i, pname in enumerate(selected_players):
            prow = player_data[player_data["player_name"] == pname].iloc[0]
            raw_values = [prow[s] for s in chosen_stats]

            # Percentile rank within the filtered population
            pct_values = []
            for s, raw in zip(chosen_stats, raw_values):
                col_data = filtered[s].dropna()
                if len(col_data) > 0:
                    pct = (col_data < raw).sum() / len(col_data) * 100
                else:
                    pct = 50
                pct_values.append(round(pct, 1))

            # Build hover text with raw + percentile
            hover = [
                f"{lab}<br>Raw: {raw:.3f}<br>Percentile: {pct:.0f}th"
                for lab, raw, pct in zip(stat_labels, raw_values, pct_values)
            ]

            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]

            fig.add_trace(go.Scatterpolar(
                r=pct_values + [pct_values[0]],  # close the polygon
                theta=stat_labels + [stat_labels[0]],
                name=pname,
                fill='toself',
                fillcolor=hex_to_rgba(color),
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertext=hover + [hover[0]],
                hoverinfo="text",
            ))

        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    ticktext=["20th", "40th", "60th", "80th", "100th"],
                    gridcolor="#334155",
                    tickfont=dict(size=10, color="#64748b"),
                    linecolor="#334155",
                ),
                angularaxis=dict(
                    gridcolor="#334155",
                    tickfont=dict(size=11, color="#94a3b8", family="DM Sans"),
                    linecolor="#334155",
                ),
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=13, color="#e2e8f0", family="DM Sans"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#334155",
                borderwidth=1,
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5,
                orientation="h",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=80, r=80, t=60, b=60),
            height=520,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Raw stat comparison table ────────────────────────────────────
        st.markdown("#### Head-to-Head Stats")
        comparison_rows = []
        for s in chosen_stats:
            row = {"Stat": ALL_STATS[s]}
            for pname in selected_players:
                prow = player_data[player_data["player_name"] == pname].iloc[0]
                row[pname] = round(prow[s], 4)
            comparison_rows.append(row)
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    elif not selected_players:
        st.info("Select players from the dropdown above to compare their stat profiles.")
    elif not chosen_stats:
        st.warning("Select at least one stat to display on the radar chart.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: STATS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("#### Distribution by Cluster")

    ex_col1, ex_col2 = st.columns([1, 1])
    with ex_col1:
        stat_to_explore = st.selectbox(
            "Choose a stat",
            options=list(ALL_STATS.keys()),
            format_func=lambda x: ALL_STATS.get(x, x),
        )
    with ex_col2:
        top_n = st.slider("Top N players to show", min_value=5, max_value=50, value=20)

    if stat_to_explore:
        nice_name = ALL_STATS[stat_to_explore]

        # Box plot by cluster
        clusters_in_view = filtered["cluster_full"].dropna().unique()
        box_fig = go.Figure()
        for cl in sorted(clusters_in_view):
            cl_data = filtered[filtered["cluster_full"] == cl][stat_to_explore].dropna()
            box_fig.add_trace(go.Box(
                y=cl_data,
                name=str(cl) if cl else "Unlabeled",
                marker_color=PLAYER_COLORS[hash(str(cl)) % len(PLAYER_COLORS)],
                boxmean=True,
            ))
        box_fig.update_layout(
            title=f"{nice_name} Distribution by Cluster",
            yaxis_title=nice_name,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="DM Sans"),
            height=400,
            margin=dict(l=60, r=20, t=60, b=40),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(box_fig, use_container_width=True, config={"displayModeBar": False})

        # Top N bar chart
        top_players = filtered.nlargest(top_n, stat_to_explore)
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=top_players["player_name"],
            y=top_players[stat_to_explore],
            marker_color="#3b82f6",
            hovertemplate="%{x}<br>" + nice_name + ": %{y:.4f}<extra></extra>",
        ))
        bar_fig.update_layout(
            title=f"Top {top_n} Players — {nice_name}",
            xaxis_title="",
            yaxis_title=nice_name,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="DM Sans"),
            height=400,
            margin=dict(l=60, r=20, t=60, b=100),
            xaxis=dict(tickangle=-45, gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PLAYER TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_roster:
    st.markdown("#### Player Roster")

    display_cols = [
        "player_name", "primary_position", "broad_position", "primary_competition",
        "cluster_full", "sub_group", "role_fit_score", "role_fit_rank",
        "role_confidence", "total_events",
    ]
    display_df = filtered[display_cols].copy()
    display_df.columns = [
        "Player", "Position", "Pos. Group", "Competition",
        "Cluster", "Sub Group", "Fit Score", "Fit Rank",
        "Confidence", "Events",
    ]

    search = st.text_input("Search players", placeholder="Type a name...")
    if search:
        display_df = display_df[display_df["Player"].str.contains(search, case=False, na=False)]

    st.dataframe(
        display_df.sort_values("Fit Score", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.caption(f"Showing {len(display_df):,} players")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SIMILAR PLAYERS
# ══════════════════════════════════════════════════════════════════════════════
ADJ_STATS = [c for c in df.columns if c.endswith("_adj")]

with tab_finder:
    st.markdown("#### Find Similar Players")

    finder_col1, finder_col2 = st.columns([2, 1])
    with finder_col1:
        all_player_names = sorted(df["player_name"].dropna().unique().tolist())
        target_player = st.selectbox(
            "Select a player",
            options=[""] + all_player_names,
            label_visibility="collapsed",
            placeholder="Search for a player...",
        )
    with finder_col2:
        same_position = st.checkbox("Same position group only", value=True)

    if target_player:
        target_row = df[df["player_name"] == target_player].iloc[0]
        target_vec = target_row[ADJ_STATS].values.astype(float)

        pool = df[df["player_name"] != target_player].copy()
        if same_position:
            pool = pool[pool["broad_position"] == target_row["broad_position"]]

        pool_vecs = pool[ADJ_STATS].values.astype(float)
        distances = np.sqrt(((pool_vecs - target_vec) ** 2).sum(axis=1))
        pool = pool.copy()
        pool["_dist"] = distances
        top3 = pool.nsmallest(3, "_dist")

        # ── Player cards ──────────────────────────────────────────────────
        all_cards = [target_row] + [top3.iloc[i] for i in range(len(top3))]
        card_colors = [PLAYER_COLORS[0]] + PLAYER_COLORS[1:4]
        card_labels = ["Selected"] + [f"#{i+1} Similar" for i in range(len(top3))]

        cols = st.columns(len(all_cards))
        for i, (prow, color, badge) in enumerate(zip(all_cards, card_colors, card_labels)):
            with cols[i]:
                dist_str = "" if i == 0 else f'<div style="font-size:11px; color:#64748b; margin-top:4px;">Distance: {top3.iloc[i-1]["_dist"]:.4f}</div>'
                st.markdown(f"""
                <div class="stat-card" style="border-top: 3px solid {color};">
                    <div style="font-size:10px; text-transform:uppercase; letter-spacing:1px; color:{color}; margin-bottom:4px;">{badge}</div>
                    <div class="value" style="font-size:15px; color:{color};">{prow['player_name']}</div>
                    <div class="label" style="margin-top:6px;">{prow.get('primary_position', '—')}</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
                        {prow.get('primary_competition', '—')} · {prow.get('cluster_full', '—')}
                    </div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                        FM Role: {prow.get('fm_role', '—')}
                    </div>
                    {dist_str}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Radar chart ───────────────────────────────────────────────────
        radar_preset = st.selectbox(
            "Stat profile for radar",
            options=list(STAT_PRESETS.keys()),
            index=list(STAT_PRESETS.keys()).index("All-Around Midfielder"),
            key="finder_preset",
        )
        radar_stats = STAT_PRESETS[radar_preset]
        radar_labels = [ALL_STATS[s] for s in radar_stats]

        radar_fig = go.Figure()
        for i, (prow, color) in enumerate(zip(all_cards, card_colors)):
            raw_values = [prow[s] for s in radar_stats]
            pct_values = []
            for s, raw in zip(radar_stats, raw_values):
                col_data = df[s].dropna()
                pct = (col_data < raw).sum() / len(col_data) * 100 if len(col_data) > 0 else 50
                pct_values.append(round(pct, 1))

            hover = [
                f"{lab}<br>Raw: {raw:.3f}<br>Percentile: {pct:.0f}th"
                for lab, raw, pct in zip(radar_labels, raw_values, pct_values)
            ]

            radar_fig.add_trace(go.Scatterpolar(
                r=pct_values + [pct_values[0]],
                theta=radar_labels + [radar_labels[0]],
                name=prow["player_name"],
                fill='toself',
                fillcolor=hex_to_rgba(color),
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertext=hover + [hover[0]],
                hoverinfo="text",
            ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    ticktext=["20th", "40th", "60th", "80th", "100th"],
                    gridcolor="#334155",
                    tickfont=dict(size=10, color="#64748b"),
                    linecolor="#334155",
                ),
                angularaxis=dict(
                    gridcolor="#334155",
                    tickfont=dict(size=11, color="#94a3b8", family="DM Sans"),
                    linecolor="#334155",
                ),
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=13, color="#e2e8f0", family="DM Sans"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#334155",
                borderwidth=1,
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5,
                orientation="h",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=80, r=80, t=60, b=60),
            height=520,
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Stat comparison table ─────────────────────────────────────────
        st.markdown("#### Head-to-Head Stats")
        comp_rows = []
        for s in radar_stats:
            row = {"Stat": ALL_STATS[s]}
            for prow in all_cards:
                row[prow["player_name"]] = round(prow[s], 4)
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    else:
        st.info("Select a player above to find their three most similar peers.")
