"""
Sub-group re-clustering pipeline.
Splits positions into 8 behaviorally meaningful sub-groups,
runs K-Means within each, assigns FM role names via ideal centroid matching,
then computes role fit scores.

Sub-groups & k:
  Goalkeeper       k=2   Goalkeeper, Sweeper Keeper
  Center Back      k=3   Central Defender, Ball-Playing Defender, Stopper
  Full Back/WB     k=4   Full Back, Inverted Wing Back, Attacking Full Back, Complete Wing Back
  Defensive Mid    k=3   Anchor, Ball-Winning Midfielder, Regista
  Central Mid      k=4   Carrilero, Central Midfielder, Box-to-Box, Mezzala
  Attacking Mid    k=3   Advanced Playmaker, Shadow Striker, Trequartista
  Winger           k=4   Raumdeuter, Inverted Winger, Winger, Inside Forward
  Striker          k=6   False Nine/DLF, Poacher, Complete Forward, Advanced Forward,
                         Pressing Forward, Target Man
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

DATA_PATH  = '/Users/zachmoskowitz/Desktop/MLBA1/modeling_outputs/player_profiles.csv'
OUTPUT_DIR = '/Users/zachmoskowitz/Desktop/MLBA1/modeling_outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = plt.cm.tab10.colors

# ══════════════════════════════════════════════════════════════════════════════
# SUB-GROUP DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
SUBGROUPS = {
    'Goalkeeper': {
        'positions': ['Goalkeeper'],
        'k': 2,
        'roles': ['Goalkeeper', 'Sweeper Keeper'],
    },
    'Center Back': {
        'positions': ['Center Back', 'Right Center Back', 'Left Center Back'],
        'k': 3,
        'roles': ['Central Defender', 'Ball-Playing Defender', 'Stopper'],
    },
    'Full Back / Wing Back': {
        'positions': ['Right Back', 'Left Back', 'Right Wing Back', 'Left Wing Back'],
        'k': 4,
        'roles': ['Full Back', 'Inverted Wing Back', 'Attacking Full Back', 'Complete Wing Back'],
    },
    'Defensive Mid': {
        'positions': ['Center Defensive Midfield',
                      'Right Defensive Midfield', 'Left Defensive Midfield'],
        'k': 3,
        'roles': ['Anchor', 'Ball-Winning Midfielder', 'Regista'],
    },
    'Central Mid': {
        'positions': ['Center Midfield', 'Right Center Midfield', 'Left Center Midfield'],
        'k': 4,
        'roles': ['Carrilero', 'Central Midfielder', 'Box-to-Box', 'Mezzala'],
    },
    'Attacking Mid': {
        'positions': ['Center Attacking Midfield',
                      'Right Attacking Midfield', 'Left Attacking Midfield'],
        'k': 3,
        'roles': ['Advanced Playmaker', 'Shadow Striker', 'Trequartista'],
    },
    'Winger': {
        'positions': ['Right Wing', 'Left Wing'],
        'k': 4,
        'roles': ['Raumdeuter', 'Inverted Winger', 'Winger', 'Inside Forward'],
    },
    'Striker': {
        'positions': ['Center Forward', 'Right Center Forward',
                      'Left Center Forward', 'Secondary Striker'],
        'k': 6,
        'roles': ['False Nine / Deep-Lying Forward', 'Poacher', 'Complete Forward',
                  'Advanced Forward', 'Pressing Forward', 'Target Man'],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# IDEAL CENTROIDS (in standardised feature space: +1=high, 0=neutral, -1=low)
# Used to match K-Means clusters → FM roles via cosine similarity
# ══════════════════════════════════════════════════════════════════════════════
ALL_FEATS = [
    'pass_completion_rate_adj', 'avg_pass_length_adj', 'progressive_pass_rate_adj',
    'cross_rate_adj', 'switch_rate_adj', 'shot_assist_rate_adj',
    'avg_xg_per_shot_adj', 'shot_on_target_rate_adj',
    'dribble_success_rate_adj', 'duel_win_rate_adj', 'duel_per_100_adj',
    'carry_per_100_adj', 'under_pressure_rate_adj', 'final_third_rate_adj',
    'pass_per_100_adj', 'shot_per_100_adj', 'dribble_per_100_adj',
    'pressure_pass_rate_adj',
]

def make_ideal(specs):
    """Build ideal centroid vector from a {feature: weight} dict."""
    v = np.zeros(len(ALL_FEATS))
    for feat, w in specs.items():
        if feat in ALL_FEATS:
            v[ALL_FEATS.index(feat)] = w
    return v

IDEAL_CENTROIDS = {
    # ── Goalkeeper ────────────────────────────────────────────────────────────
    'Goalkeeper': make_ideal({
        'pass_completion_rate_adj':  +1.0,
        'avg_pass_length_adj':       -1.0,
        'progressive_pass_rate_adj': -1.0,
        'carry_per_100_adj':         -1.0,
    }),
    'Sweeper Keeper': make_ideal({
        'avg_pass_length_adj':       +1.0,
        'progressive_pass_rate_adj': +1.0,
        'carry_per_100_adj':         +1.0,
        'pass_completion_rate_adj':  -0.5,
    }),
    # ── Center Back ───────────────────────────────────────────────────────────
    'Central Defender': make_ideal({
        'duel_win_rate_adj':         +0.5,
        'duel_per_100_adj':          +0.5,
        'pass_completion_rate_adj':  +0.5,
        'progressive_pass_rate_adj':  0.0,
    }),
    'Ball-Playing Defender': make_ideal({
        'pass_completion_rate_adj':  +1.0,
        'progressive_pass_rate_adj': +1.0,
        'avg_pass_length_adj':       +0.8,
        'duel_per_100_adj':          -0.8,
    }),
    'Stopper': make_ideal({
        'duel_per_100_adj':          +1.0,
        'duel_win_rate_adj':         +1.0,
        'under_pressure_rate_adj':   +0.7,
        'pass_completion_rate_adj':  -0.5,
        'progressive_pass_rate_adj': -0.5,
    }),
    # ── Full Back / Wing Back ─────────────────────────────────────────────────
    'Full Back': make_ideal({
        'duel_per_100_adj':          +0.4,
        'pass_completion_rate_adj':  +0.6,
        'cross_rate_adj':             0.0,
        'final_third_rate_adj':      +0.3,
        'dribble_per_100_adj':       -0.5,
    }),
    'Inverted Wing Back': make_ideal({
        'switch_rate_adj':           +1.0,
        'pass_completion_rate_adj':  +0.8,
        'progressive_pass_rate_adj': +0.8,
        'cross_rate_adj':            -1.0,
        'dribble_per_100_adj':       +0.3,
    }),
    'Attacking Full Back': make_ideal({
        'cross_rate_adj':            +0.7,
        'final_third_rate_adj':      +0.8,
        'progressive_pass_rate_adj': +0.7,
        'dribble_per_100_adj':       +0.5,
        'duel_per_100_adj':          +0.3,
    }),
    'Complete Wing Back': make_ideal({
        'cross_rate_adj':            +1.0,
        'dribble_per_100_adj':       +1.0,
        'final_third_rate_adj':      +1.0,
        'carry_per_100_adj':         +0.7,
        'duel_per_100_adj':          +0.5,
    }),
    # ── Defensive Mid ─────────────────────────────────────────────────────────
    'Anchor': make_ideal({
        'duel_win_rate_adj':         +0.8,
        'pass_completion_rate_adj':  +0.7,
        'final_third_rate_adj':      -0.8,
        'duel_per_100_adj':          +0.5,
        'progressive_pass_rate_adj': -0.5,
    }),
    'Ball-Winning Midfielder': make_ideal({
        'duel_per_100_adj':          +1.0,
        'pressure_pass_rate_adj':    +1.0,
        'duel_win_rate_adj':         +0.7,
        'final_third_rate_adj':      -0.5,
        'pass_completion_rate_adj':  -0.3,
    }),
    'Regista': make_ideal({
        'pass_completion_rate_adj':  +1.0,
        'progressive_pass_rate_adj': +1.0,
        'avg_pass_length_adj':       +0.8,
        'switch_rate_adj':           +0.6,
        'duel_per_100_adj':          -0.8,
    }),
    # ── Central Mid ───────────────────────────────────────────────────────────
    'Carrilero': make_ideal({
        'switch_rate_adj':           +1.0,
        'pass_completion_rate_adj':  +0.8,
        'duel_per_100_adj':          +0.5,
        'final_third_rate_adj':      -0.7,
        'shot_per_100_adj':          -0.7,
    }),
    'Central Midfielder': make_ideal({
        'pass_completion_rate_adj':  +0.4,
        'duel_win_rate_adj':         +0.4,
        'progressive_pass_rate_adj': +0.4,
        'final_third_rate_adj':      +0.3,
        'shot_per_100_adj':          +0.2,
    }),
    'Box-to-Box': make_ideal({
        'duel_per_100_adj':          +0.8,
        'final_third_rate_adj':      +0.8,
        'shot_per_100_adj':          +0.6,
        'progressive_pass_rate_adj': +0.5,
        'dribble_per_100_adj':       +0.5,
        'under_pressure_rate_adj':   +0.5,
    }),
    'Mezzala': make_ideal({
        'progressive_pass_rate_adj': +1.0,
        'shot_assist_rate_adj':      +0.9,
        'final_third_rate_adj':      +0.9,
        'dribble_per_100_adj':       +0.7,
        'shot_per_100_adj':          +0.6,
    }),
    # ── Attacking Mid ─────────────────────────────────────────────────────────
    'Advanced Playmaker': make_ideal({
        'shot_assist_rate_adj':      +1.0,
        'progressive_pass_rate_adj': +0.9,
        'final_third_rate_adj':      +0.8,
        'shot_per_100_adj':          -0.3,
        'dribble_per_100_adj':       +0.4,
    }),
    'Shadow Striker': make_ideal({
        'shot_per_100_adj':          +0.9,
        'final_third_rate_adj':      +0.9,
        'avg_xg_per_shot_adj':       +0.8,
        'dribble_per_100_adj':       +0.5,
        'shot_assist_rate_adj':      +0.3,
    }),
    'Trequartista': make_ideal({
        'dribble_per_100_adj':       +1.0,
        'shot_per_100_adj':          +0.8,
        'avg_xg_per_shot_adj':       +0.7,
        'final_third_rate_adj':      +0.9,
        'shot_assist_rate_adj':      +0.5,
        'pass_completion_rate_adj':  -0.5,
    }),
    # ── Winger ────────────────────────────────────────────────────────────────
    'Winger': make_ideal({
        'cross_rate_adj':            +1.0,
        'final_third_rate_adj':      +0.8,
        'dribble_per_100_adj':       +0.6,
        'progressive_pass_rate_adj': +0.5,
        'switch_rate_adj':           -0.3,
    }),
    'Inside Forward': make_ideal({
        'shot_per_100_adj':          +1.0,
        'dribble_per_100_adj':       +0.9,
        'final_third_rate_adj':      +0.9,
        'cross_rate_adj':            -0.5,
        'avg_xg_per_shot_adj':       +0.6,
    }),
    'Inverted Winger': make_ideal({
        'switch_rate_adj':           +1.0,
        'progressive_pass_rate_adj': +0.8,
        'dribble_per_100_adj':       +0.7,
        'cross_rate_adj':            -0.7,
        'shot_assist_rate_adj':      +0.7,
    }),
    'Raumdeuter': make_ideal({
        'final_third_rate_adj':      +0.8,
        'avg_xg_per_shot_adj':       +0.9,
        'shot_on_target_rate_adj':   +0.8,
        'dribble_per_100_adj':       -0.7,
        'cross_rate_adj':            -0.5,
        'under_pressure_rate_adj':   -0.5,
    }),
    # ── Striker ───────────────────────────────────────────────────────────────
    'Poacher': make_ideal({
        'shot_per_100_adj':          +1.0,
        'avg_xg_per_shot_adj':       +1.0,
        'shot_on_target_rate_adj':   +0.9,
        'duel_per_100_adj':          -0.5,
        'dribble_per_100_adj':       -0.3,
        'progressive_pass_rate_adj': -0.5,
    }),
    'Advanced Forward': make_ideal({
        'shot_per_100_adj':          +0.8,
        'dribble_per_100_adj':       +0.7,
        'final_third_rate_adj':      +0.9,
        'under_pressure_rate_adj':   +0.7,
        'progressive_pass_rate_adj': +0.5,
    }),
    'Target Man': make_ideal({
        'duel_per_100_adj':          +1.0,
        'duel_win_rate_adj':         +0.9,
        'shot_on_target_rate_adj':   +0.6,
        'shot_per_100_adj':          +0.4,
        'dribble_per_100_adj':       -0.5,
    }),
    'Complete Forward': make_ideal({
        'dribble_per_100_adj':       +1.0,
        'shot_per_100_adj':          +0.8,
        'final_third_rate_adj':      +0.9,
        'duel_per_100_adj':          +0.5,
        'avg_xg_per_shot_adj':       +0.5,
    }),
    'Pressing Forward': make_ideal({
        'duel_per_100_adj':          +0.8,
        'under_pressure_rate_adj':   +1.0,
        'shot_per_100_adj':          +0.7,
        'duel_win_rate_adj':         +0.5,
        'final_third_rate_adj':      +0.7,
    }),
    'False Nine / Deep-Lying Forward': make_ideal({
        'shot_assist_rate_adj':      +0.9,
        'progressive_pass_rate_adj': +0.9,
        'pass_completion_rate_adj':  +0.7,
        'duel_per_100_adj':          -0.7,
        'shot_per_100_adj':          +0.3,
        'avg_xg_per_shot_adj':       +0.4,
    }),
}

# ══════════════════════════════════════════════════════════════════════════════
# ROLE FIT KPIs
# ══════════════════════════════════════════════════════════════════════════════
ROLE_KPIS = {
    'Goalkeeper':                    [('pass_completion_rate_adj', True),
                                      ('pressure_pass_rate_adj',   True)],
    'Sweeper Keeper':                [('avg_carry_dist_adj',        True),
                                      ('avg_pass_length_adj',       True),
                                      ('progressive_pass_rate_adj', True)],
    'Central Defender':              [('duel_win_rate_adj',         True),
                                      ('duel_per_100_adj',          True),
                                      ('pass_completion_rate_adj',  True)],
    'Ball-Playing Defender':         [('pass_completion_rate_adj',  True),
                                      ('progressive_pass_rate_adj', True),
                                      ('avg_pass_length_adj',       True)],
    'Stopper':                       [('duel_win_rate_adj',         True),
                                      ('duel_per_100_adj',          True),
                                      ('under_pressure_rate_adj',   True)],
    'Full Back':                     [('duel_win_rate_adj',         True),
                                      ('pass_completion_rate_adj',  True),
                                      ('progressive_pass_rate_adj', True)],
    'Inverted Wing Back':            [('switch_rate_adj',           True),
                                      ('pass_completion_rate_adj',  True),
                                      ('progressive_pass_rate_adj', True)],
    'Attacking Full Back':           [('cross_rate_adj',            True),
                                      ('final_third_rate_adj',      True),
                                      ('progressive_pass_rate_adj', True)],
    'Complete Wing Back':            [('cross_rate_adj',            True),
                                      ('dribble_success_rate_adj',  True),
                                      ('carry_per_100_adj',         True),
                                      ('final_third_rate_adj',      True)],
    'Anchor':                        [('duel_win_rate_adj',         True),
                                      ('pass_completion_rate_adj',  True),
                                      ('under_pressure_rate_adj',   True)],
    'Ball-Winning Midfielder':       [('duel_win_rate_adj',         True),
                                      ('duel_per_100_adj',          True),
                                      ('pressure_pass_rate_adj',    True)],
    'Regista':                       [('pass_completion_rate_adj',  True),
                                      ('progressive_pass_rate_adj', True),
                                      ('avg_pass_length_adj',       True)],
    'Carrilero':                     [('switch_rate_adj',           True),
                                      ('pass_completion_rate_adj',  True),
                                      ('duel_win_rate_adj',         True)],
    'Central Midfielder':            [('pass_completion_rate_adj',  True),
                                      ('progressive_pass_rate_adj', True),
                                      ('duel_win_rate_adj',         True)],
    'Box-to-Box':                    [('duel_win_rate_adj',         True),
                                      ('progressive_pass_rate_adj', True),
                                      ('final_third_rate_adj',      True)],
    'Mezzala':                       [('progressive_pass_rate_adj', True),
                                      ('shot_assist_rate_adj',      True),
                                      ('final_third_rate_adj',      True)],
    'Advanced Playmaker':            [('shot_assist_rate_adj',      True),
                                      ('progressive_pass_rate_adj', True),
                                      ('final_third_rate_adj',      True)],
    'Shadow Striker':                [('shot_per_100_adj',          True),
                                      ('avg_xg_per_shot_adj',       True),
                                      ('final_third_rate_adj',      True)],
    'Trequartista':                  [('dribble_per_100_adj',       True),
                                      ('shot_per_100_adj',          True),
                                      ('avg_xg_per_shot_adj',       True)],
    'Winger':                        [('cross_rate_adj',            True),
                                      ('final_third_rate_adj',      True),
                                      ('dribble_per_100_adj',       True)],
    'Inside Forward':                [('shot_per_100_adj',          True),
                                      ('dribble_per_100_adj',       True),
                                      ('final_third_rate_adj',      True)],
    'Inverted Winger':               [('switch_rate_adj',           True),
                                      ('progressive_pass_rate_adj', True),
                                      ('shot_assist_rate_adj',      True)],
    'Raumdeuter':                    [('avg_xg_per_shot_adj',       True),
                                      ('shot_on_target_rate_adj',   True),
                                      ('final_third_rate_adj',      True)],
    'Poacher':                       [('avg_xg_per_shot_adj',       True),
                                      ('shot_on_target_rate_adj',   True),
                                      ('shot_per_100_adj',          True)],
    'Advanced Forward':              [('shot_per_100_adj',          True),
                                      ('dribble_per_100_adj',       True),
                                      ('final_third_rate_adj',      True)],
    'Target Man':                    [('duel_win_rate_adj',         True),
                                      ('duel_per_100_adj',          True),
                                      ('shot_on_target_rate_adj',   True)],
    'Complete Forward':              [('dribble_success_rate_adj',  True),
                                      ('shot_per_100_adj',          True),
                                      ('final_third_rate_adj',      True)],
    'Pressing Forward':              [('duel_per_100_adj',          True),
                                      ('under_pressure_rate_adj',   True),
                                      ('shot_per_100_adj',          True)],
    'False Nine / Deep-Lying Forward':[('shot_assist_rate_adj',    True),
                                       ('progressive_pass_rate_adj',True),
                                       ('avg_xg_per_shot_adj',      True)],
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading player profiles …")
df = pd.read_csv(DATA_PATH)
adj_cols = [c for c in ALL_FEATS if c in df.columns]
print(f"  {len(df):,} players | {len(adj_cols)} feature columns")

# Reset role columns
df['sub_group']      = ''
df['fm_role']        = ''
df['cluster_label']  = -1
df['role_fit_score'] = np.nan
df['role_fit_rank']  = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER & ASSIGN PER SUB-GROUP
# ══════════════════════════════════════════════════════════════════════════════
eval_rows    = []
all_fit_rows = []

print()
for sg_name, sg_cfg in SUBGROUPS.items():
    positions = sg_cfg['positions']
    k         = sg_cfg['k']
    roles     = sg_cfg['roles']

    sub = df[df['primary_position'].isin(positions)].copy()
    n   = len(sub)

    if n < k * 5:
        print(f"  {sg_name}: too few players ({n}) for k={k} — skipping")
        continue

    print(f"── {sg_name}  (n={n}, k={k}) ──")

    X      = sub[adj_cols].fillna(0)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # K-Means
    km     = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_sc)

    sil = silhouette_score(X_sc, labels)
    db  = davies_bouldin_score(X_sc, labels)
    ch  = calinski_harabasz_score(X_sc, labels)
    print(f"   Silhouette={sil:.3f}  DB={db:.3f}  CH={ch:.1f}")

    # ── Match clusters → FM roles via ideal centroid cosine similarity ────────
    centroids = km.cluster_centers_          # shape (k, n_feats)

    # Build ideal matrix for this sub-group's roles
    ideal_matrix = np.stack([IDEAL_CENTROIDS[r] for r in roles])   # (k, n_feats)
    # Only use adj_cols (subset of ALL_FEATS)
    feat_idx     = [ALL_FEATS.index(f) for f in adj_cols if f in ALL_FEATS]
    ideal_sub    = ideal_matrix[:, feat_idx]    # (k, len(adj_cols))

    # Cosine similarity: (k_clusters, k_roles)
    sim_matrix   = cosine_similarity(centroids, ideal_sub)

    # Optimal one-to-one assignment (maximize similarity)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    cluster_to_role  = {row_ind[i]: roles[col_ind[i]] for i in range(len(row_ind))}

    named_labels = np.array([cluster_to_role[l] for l in labels])

    # Store back
    df.loc[sub.index, 'sub_group']     = sg_name
    df.loc[sub.index, 'fm_role']       = named_labels
    df.loc[sub.index, 'cluster_label'] = labels

    # Print distribution
    for role in roles:
        cnt = (named_labels == role).sum()
        print(f"   {role:<40} n={cnt}")

    eval_rows.append({
        'Sub-group':           sg_name,
        'Players':             n,
        'k':                   k,
        'Silhouette ↑':        round(sil, 3),
        'Davies-Bouldin ↓':    round(db, 3),
        'Calinski-Harabasz ↑': round(ch, 1),
    })

    # ── Role Fit Scores ───────────────────────────────────────────────────────
    for role in roles:
        role_sub = sub[named_labels == role].copy()
        if len(role_sub) == 0:
            continue

        kpis       = ROLE_KPIS.get(role, [])
        valid_kpis = [(c, h) for c, h in kpis if c in role_sub.columns]
        kpi_cols   = [c for c, _ in valid_kpis]
        if not kpi_cols:
            continue

        for col in kpi_cols:
            role_sub[col] = role_sub[col].fillna(role_sub[col].median())

        normed = pd.DataFrame(
            MinMaxScaler().fit_transform(role_sub[kpi_cols]),
            columns=kpi_cols, index=role_sub.index
        )
        for col, higher in valid_kpis:
            if not higher:
                normed[col] = 1 - normed[col]

        role_sub['role_fit_score'] = (normed.mean(axis=1) * 100).round(1)
        role_sub['role_fit_rank']  = role_sub['role_fit_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        role_sub = role_sub.sort_values('role_fit_rank')

        df.loc[role_sub.index, 'role_fit_score'] = role_sub['role_fit_score']
        df.loc[role_sub.index, 'role_fit_rank']  = role_sub['role_fit_rank']

        top5 = role_sub.head(5)[['player_name', 'primary_competition', 'role_fit_score']].values
        print(f"   Top 5 archetypal '{role}':")
        for nm, comp, score in top5:
            print(f"     {str(nm):<42} {str(comp):<28} fit={score}")

        all_fit_rows.append(role_sub[[
            'player_name', 'primary_position', 'primary_competition',
            'sub_group', 'fm_role', 'total_events',
            'role_fit_score', 'role_fit_rank'
        ] + kpi_cols].assign(broad_position=role_sub.get('broad_position', '')))

    print()

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
eval_df = pd.DataFrame(eval_rows)
print(eval_df.to_string(index=False))
eval_df.to_csv(f'{OUTPUT_DIR}evaluation_metrics.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS — PCA scatter per sub-group
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating PCA scatter plots …")
n_sg   = len(SUBGROUPS)
ncols  = 3
nrows  = (n_sg + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
axes = axes.flatten()

for i, (sg_name, sg_cfg) in enumerate(SUBGROUPS.items()):
    ax  = axes[i]
    sub = df[df['sub_group'] == sg_name]
    if len(sub) == 0:
        ax.axis('off'); continue

    X_sc = StandardScaler().fit_transform(sub[adj_cols].fillna(0))
    X_2d = PCA(n_components=2).fit_transform(X_sc)
    roles_present = sub['fm_role'].unique()

    for j, role in enumerate(roles_present):
        mask = sub['fm_role'].values == role
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=COLORS[j % 10], alpha=0.5, s=20,
                   label=f'{role} ({mask.sum()})')

    ax.set_title(sg_name, fontweight='bold', fontsize=10)
    ax.legend(fontsize=6, ncol=1, loc='best')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('PCA Cluster View — All Sub-groups (FM Roles)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_pca_subgroups.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_pca_subgroups.png")

# ══════════════════════════════════════════════════════════════════════════════
# RADAR CHARTS — per sub-group
# ══════════════════════════════════════════════════════════════════════════════
RADAR_FEATURES = [
    ('pass_completion_rate_adj', 'Pass\nAccuracy'),
    ('progressive_pass_rate_adj','Progressive\nPassing'),
    ('cross_rate_adj',           'Crossing'),
    ('shot_per_100_adj',         'Shot\nVolume'),
    ('avg_xg_per_shot_adj',      'Shot\nQuality'),
    ('dribble_per_100_adj',      'Dribbling'),
    ('duel_win_rate_adj',        'Duels\nWon'),
    ('carry_per_100_adj',        'Ball\nCarrying'),
    ('under_pressure_rate_adj',  'Under\nPressure'),
    ('final_third_rate_adj',     'Final Third\nActivity'),
]
RADAR_FEATURES = [(f, l) for f, l in RADAR_FEATURES if f in df.columns]
radar_cols = [f for f, _ in RADAR_FEATURES]
radar_lbls = [l for _, l in RADAR_FEATURES]
N = len(radar_cols)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Normalise radar cols globally
mm = MinMaxScaler()
df_radar = df.copy()
df_radar[radar_cols] = mm.fit_transform(df_radar[radar_cols].fillna(0))

print("\nGenerating radar charts …")
for sg_name, sg_cfg in SUBGROUPS.items():
    sub   = df_radar[df_radar['sub_group'] == sg_name]
    roles = [r for r in sg_cfg['roles'] if r in sub['fm_role'].values]
    k     = len(roles)
    if k == 0: continue

    ncols_r = min(k, 3)
    nrows_r = (k + ncols_r - 1) // ncols_r
    fig     = plt.figure(figsize=(6 * ncols_r, 6 * nrows_r))
    fig.suptitle(f'{sg_name} — FM Role Profiles', fontsize=13, fontweight='bold', y=1.01)

    for c, role in enumerate(roles):
        ax  = fig.add_subplot(nrows_r, ncols_r, c + 1, polar=True)
        cdf = sub[sub['fm_role'] == role]
        if len(cdf) == 0:
            ax.set_title(f'{role}\n(no data)', size=9); continue

        vals = cdf[radar_cols].mean().tolist() + [cdf[radar_cols].mean().tolist()[0]]
        ax.plot(angles, vals, color=COLORS[c % 10], lw=2)
        ax.fill(angles, vals, color=COLORS[c % 10], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_lbls, size=7)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['', '', ''], size=5)
        ax.set_ylim(0, 1)
        ax.set_title(f'{role}\n(n={len(cdf)})', size=9, fontweight='bold', pad=14)

    plt.tight_layout()
    safe_name = sg_name.replace('/', '_').replace(' ', '_').lower()
    plt.savefig(f'{OUTPUT_DIR}05_radar_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 05_radar_{safe_name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
df.to_csv(DATA_PATH, index=False)
print(f"\nSaved: player_profiles.csv  ({len(df):,} players)")

fit_df = pd.concat(all_fit_rows, ignore_index=True)
fit_df.to_csv(f'{OUTPUT_DIR}role_fit_scores.csv', index=False)
print(f"Saved: role_fit_scores.csv")

# Cluster summary
cluster_summary = df.groupby(['sub_group', 'fm_role'])[adj_cols].mean().round(4)
cluster_summary.to_csv(f'{OUTPUT_DIR}cluster_summary.csv')
print(f"Saved: cluster_summary.csv")

print("\n" + "=" * 70)
print("DONE. 28 FM roles assigned across 8 sub-groups.")
print("=" * 70)
