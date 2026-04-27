"""
Enhancements:
  2. Updated player similarity (scoped to sub-group, cross-competition lookup)
  1. Soft assignment / confidence score (how purely a player fits their role)
  5. Feature importance per cluster (what makes each role behaviourally distinct)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

DATA_PATH  = '/Users/zachmoskowitz/Desktop/MLBA1/modeling_outputs/player_profiles.csv'
OUTPUT_DIR = '/Users/zachmoskowitz/Desktop/MLBA1/modeling_outputs/'

SUBGROUPS = {
    'Goalkeeper':            {'positions': ['Goalkeeper'],                                         'k': 2},
    'Center Back':           {'positions': ['Center Back','Right Center Back','Left Center Back'], 'k': 3},
    'Full Back / Wing Back': {'positions': ['Right Back','Left Back',
                                            'Right Wing Back','Left Wing Back'],                   'k': 4},
    'Defensive Mid':         {'positions': ['Center Defensive Midfield',
                                            'Right Defensive Midfield',
                                            'Left Defensive Midfield'],                            'k': 3},
    'Central Mid':           {'positions': ['Center Midfield',
                                            'Right Center Midfield',
                                            'Left Center Midfield'],                               'k': 4},
    'Attacking Mid':         {'positions': ['Center Attacking Midfield',
                                            'Right Attacking Midfield',
                                            'Left Attacking Midfield'],                            'k': 3},
    'Winger':                {'positions': ['Right Wing','Left Wing'],                             'k': 4},
    'Striker':               {'positions': ['Center Forward','Right Center Forward',
                                            'Left Center Forward','Secondary Striker'],             'k': 6},
}

print("Loading data …")
df = pd.read_csv(DATA_PATH)
adj_cols = [c for c in df.columns if c.endswith('_adj')]
df_out   = df.copy()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Rebuild scalers + centroids per sub-group
#          (needed for confidence score and similarity)
# ══════════════════════════════════════════════════════════════════════════════
sg_scalers    = {}   # sub_group → fitted StandardScaler
sg_km         = {}   # sub_group → fitted KMeans
sg_X_scaled   = {}   # sub_group → scaled feature matrix (all players)
sg_indices     = {}   # sub_group → df indices

print("Rebuilding scalers and K-Means per sub-group …")
for sg_name, cfg in SUBGROUPS.items():
    sub = df[df['primary_position'].isin(cfg['positions'])].copy()
    if len(sub) < cfg['k'] * 5:
        continue
    X      = sub[adj_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    km     = KMeans(n_clusters=cfg['k'], random_state=42, n_init=20)
    km.fit(X_sc)

    sg_scalers[sg_name]  = scaler
    sg_km[sg_name]       = km
    sg_X_scaled[sg_name] = X_sc
    sg_indices[sg_name]  = sub.index

# ══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 2 — Updated Player Similarity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ENHANCEMENT 2: Player Similarity (sub-group scoped)")
print("="*65)

def find_similar_players(name, top_n=10, competition_filter=None):
    """
    Find the most behaviourally similar players to `name`.

    Parameters
    ----------
    name               : str  — full or partial player name (case-insensitive)
    top_n              : int  — number of results to return
    competition_filter : str  — if set, only return players from this competition
                                (cross-competition scouting mode)

    Returns
    -------
    DataFrame of similar players with similarity score,
    or a string error message.

    Note
    ----
    Similarity is computed within the player's sub-group using cosine
    similarity on competition-adjusted, standardised features.
    High similarity = similar behavioural style, NOT similar quality.
    """
    matches = df[df['player_name'].str.contains(name, case=False, na=False)]
    if len(matches) == 0:
        return f"Player '{name}' not found."

    player_row = matches.iloc[0]
    sg         = player_row['sub_group']
    player_idx = matches.index[0]

    if sg not in sg_scalers:
        return f"Sub-group '{sg}' has no model (too few players)."

    # Position in sub-group feature matrix
    sub_indices = sg_indices[sg]
    pos_in_sg   = list(sub_indices).index(player_idx) \
                  if player_idx in sub_indices else None
    if pos_in_sg is None:
        return f"Player '{player_row['player_name']}' not found in sub-group matrix."

    X_sg     = sg_X_scaled[sg]                          # (n_sg, n_feats)
    query    = X_sg[pos_in_sg].reshape(1, -1)
    sims     = cosine_similarity(query, X_sg)[0]         # (n_sg,)

    sg_df    = df.loc[sub_indices].copy()
    sg_df['_similarity'] = sims

    # Exclude query player
    sg_df = sg_df[sg_df.index != player_idx]

    # Optional competition filter (cross-competition scouting)
    if competition_filter:
        sg_df = sg_df[
            sg_df['primary_competition'].str.contains(
                competition_filter, case=False, na=False
            )
        ]
        if len(sg_df) == 0:
            return f"No players from '{competition_filter}' found in same sub-group."

    result = (sg_df.sort_values('_similarity', ascending=False)
                   .head(top_n)
                   [['player_name', 'primary_position', 'primary_competition',
                     'fm_role', 'role_fit_score', '_similarity']]
                   .rename(columns={'_similarity': 'similarity'})
                   .copy())
    result['similarity'] = result['similarity'].round(4)

    # Header info
    print(f"\n  Query  : {player_row['player_name']}")
    print(f"  Role   : {player_row['fm_role']}")
    print(f"  League : {player_row['primary_competition']}")
    if competition_filter:
        print(f"  Filter : {competition_filter} only")
    print(f"  Sub-grp: {sg}\n")
    return result

# Demo queries
demos = [
    ('Busquets',     None,          'Same-sub-group similarity'),
    ('Messi',        None,          'Same-sub-group similarity'),
    ('Busquets',     'Bundesliga',  'Cross-competition: Bundesliga equivalent'),
    ('Harry Kane',   'Serie A',     'Cross-competition: Serie A equivalent'),
    ('Neuer',        None,          'GK same-sub-group'),
]

for name, comp, desc in demos:
    print(f"  ── {desc} ──")
    res = find_similar_players(name, top_n=5, competition_filter=comp)
    if isinstance(res, str):
        print(f"  {res}")
    else:
        print(res.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 1 — Soft Assignment / Confidence Score
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ENHANCEMENT 1: Role Confidence Score")
print("="*65)
print("  Formula: confidence = (d2 - d1) / (d2 + d1) × 100")
print("  100 = perfectly central, 0 = exactly on cluster boundary\n")

df_out['role_confidence'] = np.nan

for sg_name, cfg in SUBGROUPS.items():
    if sg_name not in sg_km:
        continue

    km      = sg_km[sg_name]
    X_sc    = sg_X_scaled[sg_name]
    indices = sg_indices[sg_name]

    # Euclidean distance to every centroid
    dists = np.linalg.norm(
        X_sc[:, np.newaxis, :] - km.cluster_centers_[np.newaxis, :, :],
        axis=2
    )   # shape (n_players, k)

    # Sort distances per player
    sorted_dists = np.sort(dists, axis=1)
    d1 = sorted_dists[:, 0]   # distance to nearest centroid
    d2 = sorted_dists[:, 1]   # distance to second nearest

    # Confidence: 0 = on boundary, 100 = perfectly central
    with np.errstate(invalid='ignore'):
        confidence = np.where(
            (d1 + d2) > 0,
            (d2 - d1) / (d2 + d1) * 100,
            100.0
        )

    df_out.loc[indices, 'role_confidence'] = np.round(confidence, 1)

# Summary stats
print("  Confidence score distribution by sub-group:\n")
print(f"  {'Sub-group':<25} {'Mean':>6}  {'Median':>6}  {'<25 (borderline)':>16}")
for sg_name in SUBGROUPS:
    sub = df_out[df_out['sub_group'] == sg_name]['role_confidence'].dropna()
    if len(sub) == 0: continue
    borderline_pct = (sub < 25).mean() * 100
    print(f"  {sg_name:<25} {sub.mean():>6.1f}  {sub.median():>6.1f}  {borderline_pct:>14.1f}%")

# Flag borderline players (confidence < 25 = within bottom quartile)
borderline = df_out[df_out['role_confidence'] < 25][
    ['player_name', 'sub_group', 'fm_role', 'role_confidence', 'primary_competition']
].sort_values('role_confidence')
print(f"\n  {len(borderline)} borderline players (confidence < 25):")
print(borderline.head(20).to_string(index=False))

# ── Confidence distribution plot ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, sg_name in enumerate(SUBGROUPS):
    ax  = axes[i]
    sub = df_out[df_out['sub_group'] == sg_name]['role_confidence'].dropna()
    ax.hist(sub, bins=20, color='#2196F3', edgecolor='k', linewidth=0.4, alpha=0.8)
    ax.axvline(25,  color='red',    ls='--', lw=1.5, label='Borderline (<25)')
    ax.axvline(sub.median(), color='orange', ls='-', lw=1.5,
               label=f'Median={sub.median():.0f}')
    ax.set_title(sg_name, fontweight='bold', fontsize=9)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('# Players')
    ax.legend(fontsize=7)

plt.suptitle('Role Confidence Score Distribution by Sub-group\n'
             '(100=perfectly archetypal, 0=on cluster boundary)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}08_confidence_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: 08_confidence_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 5 — Feature Importance Per Cluster
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ENHANCEMENT 5: Feature Importance Per Role")
print("="*65)
print("  Method: z-score of cluster mean vs sub-group mean")
print("  +ve = distinctively HIGH for this role")
print("  -ve = distinctively LOW for this role\n")

FEAT_LABELS = {
    'pass_completion_rate_adj':  'Pass Accuracy',
    'avg_pass_length_adj':       'Pass Length',
    'progressive_pass_rate_adj': 'Progressive Passing',
    'cross_rate_adj':            'Crossing',
    'switch_rate_adj':           'Switching',
    'shot_assist_rate_adj':      'Shot Assists',
    'avg_xg_per_shot_adj':       'xG per Shot',
    'shot_on_target_rate_adj':   'Shot Accuracy',
    'dribble_success_rate_adj':  'Dribble Success',
    'duel_win_rate_adj':         'Duel Win Rate',
    'duel_per_100_adj':          'Duel Volume',
    'carry_per_100_adj':         'Carry Volume',
    'under_pressure_rate_adj':   'Actions Under Pressure',
    'final_third_rate_adj':      'Final Third Activity',
    'pass_per_100_adj':          'Pass Volume',
    'shot_per_100_adj':          'Shot Volume',
    'dribble_per_100_adj':       'Dribble Volume',
    'pressure_pass_rate_adj':    'Passing Under Pressure',
}

importance_rows = []

for sg_name, cfg in SUBGROUPS.items():
    sub = df_out[df_out['sub_group'] == sg_name]
    if len(sub) == 0: continue

    sg_mean = sub[adj_cols].mean()
    sg_std  = sub[adj_cols].std().replace(0, 1e-9)
    roles   = sub['fm_role'].unique()

    for role in roles:
        role_sub  = sub[sub['fm_role'] == role]
        role_mean = role_sub[adj_cols].mean()
        z_scores  = (role_mean - sg_mean) / sg_std

        # Top 3 positive (most distinctively high)
        top_pos = z_scores.nlargest(3)
        # Top 3 negative (most distinctively low)
        top_neg = z_scores.nsmallest(3)

        print(f"  {role} (sub-group: {sg_name}):")
        print(f"    HIGH: " +
              ", ".join([f"{FEAT_LABELS.get(f, f)} ({v:+.2f}σ)"
                         for f, v in top_pos.items()]))
        print(f"    LOW:  " +
              ", ".join([f"{FEAT_LABELS.get(f, f)} ({v:+.2f}σ)"
                         for f, v in top_neg.items()]))

        for feat in adj_cols:
            importance_rows.append({
                'sub_group':   sg_name,
                'fm_role':     role,
                'feature':     feat,
                'feature_label': FEAT_LABELS.get(feat, feat),
                'z_score':     round(z_scores[feat], 3),
                'n_players':   len(role_sub),
            })

importance_df = pd.DataFrame(importance_rows)
importance_df.to_csv(f'{OUTPUT_DIR}feature_importance_per_role.csv', index=False)
print(f"\n  Saved: feature_importance_per_role.csv")

# ── Heatmap: z-scores per role per sub-group ──────────────────────────────────
print("\n  Generating feature importance heatmaps …")

for sg_name, cfg in SUBGROUPS.items():
    sg_imp = importance_df[importance_df['sub_group'] == sg_name]
    if len(sg_imp) == 0: continue

    pivot = sg_imp.pivot(index='feature_label', columns='fm_role', values='z_score')

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label='Z-score vs sub-group mean')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                        fontsize=7,
                        color='black' if abs(val) < 1.5 else 'white')

    ax.set_title(f'{sg_name} — Feature Importance by FM Role\n'
                 f'(z-score vs sub-group mean, green=high, red=low)',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()
    safe = sg_name.replace('/', '_').replace(' ', '_').lower()
    plt.savefig(f'{OUTPUT_DIR}09_importance_{safe}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 09_importance_{safe}.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE UPDATED PLAYER PROFILES
# ══════════════════════════════════════════════════════════════════════════════
df_out.to_csv(DATA_PATH, index=False)
print(f"\nSaved: player_profiles.csv  (+ role_confidence column)")
print("\n" + "="*65)
print("DONE.")
print("="*65)
