import argparse
import os
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_colwidth', None)

DEFAULT_DATA_PATH = 'data/hea_dataset.csv'
DEFAULT_OUT_DIR = 'outputs'
DEFAULT_N_TRIALS = 80
DEFAULT_N_BOOT = 40
RANDOM_SEED = 42

TARGET_COL = 'PROPERTY: YS (MPa)'
TEMP_COL = r'PROPERTY: Test temperature ($^\circ$C)'
GS_COL = r'PROPERTY: grain size ($\mu$m)'
REQUIRED_ELEMENTS = ['Al', 'Cr', 'Co', 'Ni', 'Fe']
SCREEN_ELEMENTS = ['Al', 'Cr', 'Co', 'Ni', 'Fe', 'Mo', 'Ti', 'Cu']

ELEM_PROPS = {
    'Ag': (1.445, 11, 1.93, 1234.9, 30),  'Al': (1.432, 3,  1.61, 933.5,  26),
    'As': (1.245, 5,  2.18, 1090.0, 8),   'Au': (1.442, 11, 2.54, 1337.3, 27),
    'B':  (0.820, 3,  2.04, 2349.0, 200), 'Ba': (2.174, 2,  0.89, 1000.0, 4.9),
    'Be': (1.128, 2,  1.57, 1560.0, 132), 'Bi': (1.545, 5,  2.02, 544.7,  12),
    'C':  (0.773, 4,  2.55, 3800.0, 200), 'Ca': (1.974, 2,  1.00, 1115.0, 7.4),
    'Cd': (1.489, 12, 1.69, 594.2,  19),  'Ce': (1.825, 3,  1.12, 1068.0, 14),
    'Co': (1.251, 9,  1.88, 1768.0, 75),  'Cr': (1.249, 6,  1.66, 2180.0, 115),
    'Cs': (2.655, 1,  0.79, 301.6,  1.7), 'Cu': (1.278, 11, 1.90, 1357.8, 48),
    'Dy': (1.773, 3,  1.22, 1680.0, 25),  'Er': (1.757, 3,  1.24, 1802.0, 28),
    'Eu': (1.984, 2,  1.20, 1099.0, 7.9), 'Fe': (1.241, 8,  1.83, 1811.0, 82),
    'Ga': (1.392, 3,  1.81, 302.9,  37),  'Gd': (1.801, 3,  1.20, 1585.0, 22),
    'Ge': (1.240, 4,  2.01, 1211.4, 56),  'Hf': (1.578, 4,  1.30, 2506.0, 30),
    'Ho': (1.766, 3,  1.23, 1734.0, 26),  'In': (1.663, 3,  1.78, 429.8,  3.8),
    'Ir': (1.357, 9,  2.20, 2719.0, 210), 'K':  (2.272, 1,  0.82, 336.5,  1.3),
    'La': (1.879, 3,  1.10, 1193.0, 14),  'Li': (1.519, 1,  0.98, 453.7,  4.2),
    'Lu': (1.735, 3,  1.27, 1925.0, 27),  'Mg': (1.601, 2,  1.31, 923.0,  17),
    'Mn': (1.350, 7,  1.55, 1519.0, 80),  'Mo': (1.363, 6,  2.16, 2896.0, 126),
    'N':  (0.750, 5,  3.04, 63.2,   0),   'Na': (1.858, 1,  0.93, 370.9,  3.3),
    'Nb': (1.429, 5,  1.60, 2750.0, 38),  'Nd': (1.821, 3,  1.14, 1297.0, 16),
    'Ni': (1.246, 10, 1.91, 1728.0, 76),  'Os': (1.352, 8,  2.20, 3306.0, 222),
    'P':  (1.060, 5,  2.19, 317.3,  3),   'Pa': (1.606, 5,  1.50, 1841.0, 40),
    'Pb': (1.750, 4,  2.33, 600.6,  5.6), 'Pd': (1.375, 10, 2.20, 1828.1, 44),
    'Pm': (1.811, 3,  1.13, 1315.0, 18),  'Pr': (1.828, 3,  1.13, 1208.0, 15),
    'Pt': (1.387, 10, 2.28, 2041.4, 61),  'Pu': (1.592, 8,  1.28, 912.5,  43),
    'Rb': (2.475, 1,  0.82, 312.5,  1.1), 'Re': (1.375, 7,  1.90, 3459.0, 178),
    'Rh': (1.345, 9,  2.28, 2237.0, 150), 'Ru': (1.338, 8,  2.20, 2607.0, 173),
    'Sb': (1.450, 5,  2.05, 903.8,  20),  'Sc': (1.641, 3,  1.36, 1814.0, 29),
    'Si': (1.153, 4,  1.90, 1687.0, 60),  'Sm': (1.802, 3,  1.17, 1345.0, 13),
    'Sn': (1.620, 4,  1.96, 505.1,  18),  'Sr': (2.151, 2,  0.95, 1050.0, 6.1),
    'Ta': (1.430, 5,  1.50, 3290.0, 69),  'Tb': (1.782, 3,  1.20, 1629.0, 23),
    'Tc': (1.358, 7,  1.90, 2430.0, 123), 'Te': (1.432, 6,  2.10, 722.7,  16),
    'Th': (1.798, 4,  1.30, 2023.0, 31),  'Ti': (1.462, 4,  1.54, 1941.0, 44),
    'Tl': (1.700, 3,  1.62, 577.0,  2.8), 'Tm': (1.746, 3,  1.25, 1818.0, 29),
    'U':  (1.542, 6,  1.38, 1405.3, 74),  'V':  (1.316, 5,  1.63, 2183.0, 47),
    'W':  (1.367, 6,  2.36, 3695.0, 161), 'Y':  (1.801, 3,  1.22, 1799.0, 26),
    'Yb': (1.940, 2,  1.10, 1097.0, 9.9), 'Zn': (1.332, 12, 1.65, 692.7,  43),
    'Zr': (1.603, 4,  1.33, 2128.0, 33)
}

DH_PAIRS = {
    ('Al', 'C'): -36,  ('Al', 'Co'): -19, ('Al', 'Cr'): -10, ('Al', 'Cu'): -1,
    ('Al', 'Fe'): -11, ('Al', 'Mn'): -19, ('Al', 'Mo'): -5,  ('Al', 'Nb'): -18,
    ('Al', 'Ni'): -22, ('Al', 'Si'): -19, ('Al', 'Ti'): -30, ('Al', 'V'): -16,
    ('C', 'Co'): -42,  ('C', 'Cr'): -61,  ('C', 'Cu'): 33,   ('C', 'Fe'): -50,
    ('C', 'Mn'): -66,  ('C', 'Mo'): -67,  ('C', 'Nb'): -102, ('C', 'Ni'): -39,
    ('C', 'Si'): -39,  ('C', 'Ti'): -109, ('C', 'V'): -82,
    ('Co', 'Cr'): -4,  ('Co', 'Cu'): 6,   ('Co', 'Fe'): -1,  ('Co', 'Mn'): -5,
    ('Co', 'Mo'): -5,  ('Co', 'Nb'): -25, ('Co', 'Ni'): 0,   ('Co', 'Si'): -38,
    ('Co', 'Ti'): -28, ('Co', 'V'): -14,
    ('Cr', 'Cu'): 12,  ('Cr', 'Fe'): -1,  ('Cr', 'Mn'): 2,   ('Cr', 'Mo'): 0,
    ('Cr', 'Nb'): -7,  ('Cr', 'Ni'): -7,  ('Cr', 'Si'): -37, ('Cr', 'Ti'): -7,  ('Cr', 'V'): -2,
    ('Cu', 'Fe'): 13,  ('Cu', 'Mn'): 4,   ('Cu', 'Mo'): 19,  ('Cu', 'Nb'): 3,
    ('Cu', 'Ni'): 4,   ('Cu', 'Si'): -19, ('Cu', 'Ti'): -9,  ('Cu', 'V'): 5,
    ('Fe', 'Mn'): 0,   ('Fe', 'Mo'): -2,  ('Fe', 'Nb'): -16, ('Fe', 'Ni'): -2,
    ('Fe', 'Si'): -35, ('Fe', 'Ti'): -17, ('Fe', 'V'): -7,
    ('Mn', 'Mo'): 5,   ('Mn', 'Nb'): -4,  ('Mn', 'Ni'): -8,  ('Mn', 'Si'): -45,
    ('Mn', 'Ti'): -8,  ('Mn', 'V'): -1,
    ('Mo', 'Nb'): -6,  ('Mo', 'Ni'): -7,  ('Mo', 'Si'): -35, ('Mo', 'Ti'): -4,  ('Mo', 'V'): 0,
    ('Nb', 'Ni'): -30, ('Nb', 'Si'): -56, ('Nb', 'Ti'): 2,   ('Nb', 'V'): -1,
    ('Ni', 'Si'): -40, ('Ni', 'Ti'): -35, ('Ni', 'V'): -18,
    ('Si', 'Ti'): -66, ('Si', 'V'): -48,
    ('Ti', 'V'): -2
}

ELEM_DENSITY = {
    'Al': 2.70, 'Cr': 7.19, 'Co': 8.90, 'Ni': 8.91,
    'Fe': 7.87, 'Mo': 10.28, 'Ti': 4.51, 'Cu': 8.96
}


class YSPredictor:
    def __init__(self, model, feature_cols, cat_features, active_elements, enthalpy_matrix):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.cat_features = list(cat_features)
        self.active_elements = list(active_elements)
        self.enthalpy_matrix = np.asarray(enthalpy_matrix, dtype=float)
        self.prop_vectors = np.array([ELEM_PROPS[el] for el in self.active_elements]).T

    def build_feature_df(
        self,
        comp_at_pct,
        temp_c=25.0,
        test_type='T',
        processing='CAST',
        phase_type='BCC',
        grain_size_um=np.nan
    ):
        row = {col: np.nan for col in self.feature_cols}
        for col in self.cat_features:
            row[col] = 'Unknown'

        if TEMP_COL in row:
            row[TEMP_COL] = float(temp_c)
        if 'PROPERTY: Type of test' in row:
            row['PROPERTY: Type of test'] = test_type
        if 'PROPERTY: Processing method' in row:
            row['PROPERTY: Processing method'] = processing
        if 'PROPERTY: BCC/FCC/other' in row:
            row['PROPERTY: BCC/FCC/other'] = phase_type
        if GS_COL in row:
            row[GS_COL] = grain_size_um
        if 'DESC: Hall_Petch_d_inv_sqrt' in row:
            row['DESC: Hall_Petch_d_inv_sqrt'] = (
                1.0 / np.sqrt(grain_size_um) if pd.notna(grain_size_um) and grain_size_um > 0 else np.nan
            )

        for p_col in [col for col in self.feature_cols if col.startswith('PHASE:')]:
            row[p_col] = 1.0 if p_col == f'PHASE: {phase_type}' else 0.0

        total = sum(comp_at_pct.values())
        c_vec = np.array([comp_at_pct.get(el, 0.0) / total for el in self.active_elements])
        for el, frac in zip(self.active_elements, c_vec):
            row[el] = frac * 100.0

        r, vec, chi, tm, g = self.prop_vectors
        r_bar, vec_bar, chi_bar, tm_bar, g_bar = c_vec @ r, c_vec @ vec, c_vec @ chi, c_vec @ tm, c_vec @ g
        c_safe = np.where(c_vec > 0, c_vec, 1.0)
        ds_val = -8.314 * np.sum(np.where(c_vec > 0, c_vec * np.log(c_safe), 0.0))
        dh_val = 2.0 * float((c_vec @ self.enthalpy_matrix) @ c_vec)

        r_denom = r_bar if r_bar != 0 else 1.0
        g_denom = g_bar if g_bar != 0 else 1.0

        row['DESC: VEC'] = vec_bar
        row['DESC: delta_r (%)'] = np.sqrt(np.sum(c_vec * ((1.0 - r / r_denom) ** 2))) * 100.0
        row['DESC: delta_G (%)'] = np.sqrt(np.sum(c_vec * ((1.0 - g / g_denom) ** 2))) * 100.0
        row['DESC: delta_chi'] = np.sqrt(np.sum(c_vec * ((chi - chi_bar) ** 2)))
        row['DESC: dS_mix'] = ds_val
        row['DESC: Tm_avg (K)'] = tm_bar
        row['DESC: G_avg (GPa)'] = g_bar
        row['DESC: Num_Elements'] = int((c_vec > 0.005).sum())
        row['DESC: dH_mix (kJ/mol)'] = dh_val
        row['DESC: Omega'] = (tm_bar * ds_val) / (abs(dh_val) * 1000.0 + 1e-6)

        return pd.DataFrame([row])[self.feature_cols]

    def predict(
        self,
        comp_at_pct,
        temp_c=25.0,
        test_type='T',
        processing='CAST',
        phase_type='BCC',
        grain_size_um=np.nan
    ):
        x_cand = self.build_feature_df(comp_at_pct, temp_c, test_type, processing, phase_type, grain_size_um)
        return float(self.model.predict(x_cand)[0])

    def save(self, path):
        joblib.dump(
            {
                'model': self.model,
                'feature_cols': self.feature_cols,
                'cat_features': self.cat_features,
                'active_elements': self.active_elements,
                'enthalpy_matrix': self.enthalpy_matrix,
            },
            path
        )

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        return cls(
            model=payload['model'],
            feature_cols=payload['feature_cols'],
            cat_features=payload['cat_features'],
            active_elements=payload['active_elements'],
            enthalpy_matrix=payload['enthalpy_matrix'],
        )


def build_enthalpy_matrix(elements):
    n_el = len(elements)
    h_mat = np.zeros((n_el, n_el))
    for i, el1 in enumerate(elements):
        for j, el2 in enumerate(elements):
            if i != j:
                h_mat[i, j] = DH_PAIRS.get((el1, el2), DH_PAIRS.get((el2, el1), 0.0))
    return h_mat


def load_and_prepare_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Provide a CSV containing elemental columns (at.%), "
            f"'{TARGET_COL}', 'PROPERTY: Type of test', 'PROPERTY: Processing method', "
            f"and '{TEMP_COL}'."
        )

    df = pd.read_csv(data_path, low_memory=False)
    element_cols = list(df.columns[:74])

    df_ys = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()
    df_sub = df_ys[(df_ys[REQUIRED_ELEMENTS].fillna(0) > 0).all(axis=1)].copy()

    active_elements = [el for el in element_cols if df_sub[el].fillna(0).sum() > 0]
    zero_elements = [el for el in element_cols if el not in active_elements]

    leaky_or_id_cols = [
        'REFERENCE: doi',
        'PROPERTY: UTS (MPa)',
        'PROPERTY: HV',
        'PROPERTY: Elongation (%)',
        'PROPERTY: Elongation plastic (%)',
        'PROPERTY: Exp. Young modulus (GPa)',
        'PROPERTY: Exp. Density (g/cm$^3$)'
    ]
    df_sub = df_sub.drop(columns=zero_elements + [c for c in leaky_or_id_cols if c in df_sub.columns])

    cat_cols = ['PROPERTY: BCC/FCC/other', 'PROPERTY: Processing method', 'PROPERTY: Type of test']
    for col in cat_cols:
        if col in df_sub.columns:
            df_sub[col] = df_sub[col].fillna('Unknown').astype(str).str.strip()

    h_mat = build_enthalpy_matrix(active_elements)
    comp = df_sub[active_elements].fillna(0.0).values
    row_sums = np.where(comp.sum(axis=1, keepdims=True) == 0, 1.0, comp.sum(axis=1, keepdims=True))
    c = comp / row_sums

    r, vec, chi, tm, g = np.array([ELEM_PROPS[el] for el in active_elements]).T
    r_bar, vec_bar, chi_bar, tm_bar, g_bar = c @ r, c @ vec, c @ chi, c @ tm, c @ g
    c_safe = np.where(c > 0, c, 1.0)
    ds_mix = -8.314 * np.sum(np.where(c > 0, c * np.log(c_safe), 0.0), axis=1)
    dh_mix = 2.0 * np.sum((c @ h_mat) * c, axis=1)

    df_sub['DESC: VEC'] = vec_bar
    df_sub['DESC: delta_r (%)'] = np.sqrt(np.sum(c * ((1.0 - r[None, :] / r_bar[:, None]) ** 2), axis=1)) * 100.0
    df_sub['DESC: delta_G (%)'] = np.sqrt(np.sum(c * ((1.0 - g[None, :] / np.where(g_bar[:, None] == 0, 1, g_bar[:, None])) ** 2), axis=1)) * 100.0
    df_sub['DESC: delta_chi'] = np.sqrt(np.sum(c * ((chi[None, :] - chi_bar[:, None]) ** 2), axis=1))
    df_sub['DESC: dS_mix'] = ds_mix
    df_sub['DESC: Tm_avg (K)'] = tm_bar
    df_sub['DESC: G_avg (GPa)'] = g_bar
    df_sub['DESC: Num_Elements'] = (c > 0.005).sum(axis=1)
    df_sub['DESC: dH_mix (kJ/mol)'] = dh_mix
    df_sub['DESC: Omega'] = (tm_bar * ds_mix) / (np.abs(dh_mix) * 1000.0 + 1e-6)

    if GS_COL in df_sub.columns:
        df_sub['DESC: Hall_Petch_d_inv_sqrt'] = np.where(
            df_sub[GS_COL].notna() & (df_sub[GS_COL] > 0),
            1.0 / np.sqrt(df_sub[GS_COL]),
            np.nan
        )

    all_nan_cols = [col for col in df_sub.columns if df_sub[col].isna().all()]
    x_xgb = df_sub.drop(columns=[TARGET_COL] + all_nan_cols)
    y_xgb = df_sub[TARGET_COL].values
    comp_groups = df_sub[active_elements].fillna(0).round(2).astype(str).agg('_'.join, axis=1)

    cat_features = [col for col in x_xgb.select_dtypes(include=['object', 'string']).columns]
    num_features = [col for col in x_xgb.columns if col not in cat_features]

    return df_sub, x_xgb, y_xgb, comp_groups, active_elements, cat_features, num_features, h_mat


def tune_and_train_xgb(x_xgb, y_xgb, comp_groups, cat_features, num_features, n_trials=DEFAULT_N_TRIALS, seed=RANDOM_SEED):
    prep_xgb = ColumnTransformer([
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ])
    gkf = GroupKFold(n_splits=5)

    def objective_xgb(trial):
        params = {
            'objective': trial.suggest_categorical('objective', ['reg:squarederror', 'reg:absoluteerror']),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'random_state': seed,
            'n_jobs': -1
        }
        pipe = Pipeline([
            ('prep', clone(prep_xgb)),
            ('xgb', XGBRegressor(**params))
        ])
        scores = cross_val_score(
            pipe, x_xgb, y_xgb,
            groups=comp_groups, cv=gkf,
            scoring='neg_root_mean_squared_error'
        )
        return -scores.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)

    best_xgb = Pipeline([
        ('prep', clone(prep_xgb)),
        ('xgb', XGBRegressor(**study.best_params, random_state=seed, n_jobs=-1))
    ])

    oof_xgb = cross_val_predict(best_xgb, x_xgb, y_xgb, groups=comp_groups, cv=gkf)
    best_xgb.fit(x_xgb, y_xgb)

    return best_xgb, oof_xgb, study.best_params


def export_diagnostics(best_xgb, x_xgb, y_xgb, oof_xgb, df_sub, out_dir, show_plots=False):
    temps = df_sub[TEMP_COL].fillna(25.0).values if TEMP_COL in df_sub.columns else np.full(len(df_sub), 25.0)
    rt_mask = temps <= 30.0

    print(f"Overall OOF R2   : {r2_score(y_xgb, oof_xgb):.4f} | MAE: {mean_absolute_error(y_xgb, oof_xgb):.1f} MPa | RMSE: {root_mean_squared_error(y_xgb, oof_xgb):.1f} MPa")
    print(f"Room Temp (n={rt_mask.sum()}) : R2 = {r2_score(y_xgb[rt_mask], oof_xgb[rt_mask]):.4f} | MAE = {mean_absolute_error(y_xgb[rt_mask], oof_xgb[rt_mask]):.1f} MPa")

    feat_names = [name.split('__')[-1] for name in best_xgb.named_steps['prep'].get_feature_names_out()]
    importances = best_xgb.named_steps['xgb'].feature_importances_

    fi_xgb = (
        pd.DataFrame({'Feature': feat_names, 'Importance (%)': (importances * 100).round(2)})
        .sort_values('Importance (%)', ascending=False)
        .reset_index(drop=True)
    )
    print("\nTop 15 Feature Importances:")
    print(fi_xgb.head(15).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    max_val = max(y_xgb.max(), oof_xgb.max()) * 1.05
    axes[0].scatter(y_xgb[rt_mask], oof_xgb[rt_mask], alpha=0.75, label='Room Temp (<=30 C)', edgecolors='k', linewidth=0.5)
    axes[0].scatter(y_xgb[~rt_mask], oof_xgb[~rt_mask], alpha=0.85, marker='s', color='#dd6b20', label='High Temp (>30 C)', edgecolors='k', linewidth=0.5)
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=1.5)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_xlabel('Experimental Yield Strength (MPa)')
    axes[0].set_ylabel('Out-of-Fold Predicted YS (MPa)')
    axes[0].set_title(f'XGBoost OOF Parity Plot (OOF R2 = {r2_score(y_xgb, oof_xgb):.3f})')
    axes[0].legend()

    top_15 = fi_xgb.head(15).iloc[::-1]
    axes[1].barh(top_15['Feature'], top_15['Importance (%)'], color='#2f855a')
    axes[1].set_xlabel('Relative Importance (%)')
    axes[1].set_title('Top 15 Feature Importances (Optuna XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'oof_parity_and_importance.png'), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    prep_step = best_xgb.named_steps['prep']
    xgb_step = best_xgb.named_steps['xgb']
    x_trans_df = pd.DataFrame(prep_step.transform(x_xgb), columns=feat_names)

    explainer = shap.TreeExplainer(xgb_step)
    shap_values = explainer(x_trans_df)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_trans_df, max_display=15, show=False)
    plt.title('SHAP Impact on HEA Yield Strength (Top 15 Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def screen_compressive_candidates(predictor, n_samples=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    candidates = []

    for _ in range(n_samples):
        core = rng.uniform(10, 30, size=5)
        add_mask = rng.choice([0.0, 1.0], size=3, p=[0.4, 0.6])
        adds = rng.uniform(0, 10, size=3) * add_mask

        raw = np.concatenate([core, adds])
        at_pct = (raw / raw.sum()) * 100.0
        comp_dict = {el: round(val, 2) for el, val in zip(SCREEN_ELEMENTS, at_pct) if val > 0.1}

        rho = sum((comp_dict[el] / 100.0) * ELEM_DENSITY[el] for el in comp_dict)
        vec_val = sum((comp_dict[el] / 100.0) * ELEM_PROPS[el][1] for el in comp_dict)
        phase = 'BCC' if vec_val < 6.87 else ('FCC' if vec_val >= 8.0 else 'Multi-phase')

        pred_ys = predictor.predict(comp_dict, temp_c=25.0, test_type='C', processing='CAST', phase_type=phase)
        formula = " ".join(f"{el}{val:.1f}" for el, val in comp_dict.items())
        candidates.append({
            'Composition (at.%)': formula,
            'VEC': round(vec_val, 2),
            'Est. Phase': phase,
            'Density (g/cm3)': round(rho, 2),
            'Pred. Comp. YS (MPa)': round(pred_ys, 1),
            'Specific YS (kN*m/kg)': round(pred_ys / rho, 1)
        })

    screen_df = pd.DataFrame(candidates)
    top10 = screen_df.sort_values('Specific YS (kN*m/kg)', ascending=False).head(10).reset_index(drop=True)
    return screen_df, top10


def plot_screening_pareto(screen_df, top10, out_dir, show_plots=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for phase, grp in screen_df.groupby('Est. Phase'):
        ax.scatter(grp['VEC'], grp['Specific YS (kN*m/kg)'], alpha=0.55, label=phase, edgecolors='none')

    ax.scatter(top10['VEC'], top10['Specific YS (kN*m/kg)'], color='red', s=70, edgecolors='black', zorder=5, label='Top 10 Specific YS')
    ax.axvline(6.87, color='gray', linestyle='--', lw=1.2, label='BCC / Dual-Phase Boundary (VEC = 6.87)')
    ax.axvline(8.00, color='black', linestyle=':', lw=1.2, label='FCC Stability Threshold (VEC = 8.0)')
    ax.set_xlabel('Valence Electron Concentration (VEC)')
    ax.set_ylabel('Predicted Specific Yield Strength (kN*m/kg)')
    ax.set_title('Virtual Screening: Strength-to-Weight vs. VEC Phase Stability')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'screening_ys_vs_vec.png'), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)


def parse_formula(formula):
    return {el: float(val) for el, val in re.findall(r'([A-Z][a-z]?)([0-9.]+)', formula)}


def format_formula(row, active_elements):
    return " ".join(f"{el}{row[el]:.1f}" for el in active_elements if row[el] > 0)


def evaluate_bootstrap_uncertainty(predictor, df_sub, x_xgb, y_xgb, comp_groups, top10, n_boot=DEFAULT_N_BOOT):
    boot_models = []
    for seed in range(n_boot):
        gss_b = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        b_idx, _ = next(gss_b.split(x_xgb, y_xgb, groups=comp_groups))
        m_b = clone(predictor.model)
        m_b.fit(x_xgb.iloc[b_idx], y_xgb[b_idx])
        boot_models.append(m_b)

    active_elements = predictor.active_elements
    exp_comp_matrix = df_sub[active_elements].fillna(0.0).values
    validation_rows = []

    for _, r_cand in top10.iterrows():
        c_dict = parse_formula(r_cand['Composition (at.%)'])
        cand_vec = np.array([c_dict.get(el, 0.0) for el in active_elements])

        dists = np.linalg.norm(exp_comp_matrix - cand_vec[None, :], axis=1)
        nn_idx = int(np.argmin(dists))
        nn_row = df_sub.iloc[nn_idx]

        x_row = predictor.build_feature_df(
            c_dict, temp_c=25.0, test_type='C', processing='CAST', phase_type=r_cand['Est. Phase']
        )
        b_preds = np.array([m.predict(x_row)[0] for m in boot_models])

        validation_rows.append({
            'Candidate Composition': r_cand['Composition (at.%)'],
            'Pred. YS (MPa)': r_cand['Pred. Comp. YS (MPa)'],
            'Boot Std (MPa)': round(float(b_preds.std()), 1),
            '90% CI (MPa)': f"{np.percentile(b_preds, 5):.0f} - {np.percentile(b_preds, 95):.0f}",
            'NN Dist (at.%)': round(float(dists[nn_idx]), 1),
            'Closest Exp. Alloy': format_formula(nn_row, active_elements),
            'Exp. YS (MPa)': round(float(nn_row[TARGET_COL]), 1),
            'Exp. Test/Proc': f"{nn_row['PROPERTY: Type of test']} / {nn_row['PROPERTY: Processing method']}"
        })

    return pd.DataFrame(validation_rows)


def screen_tensile_high_vec(predictor, n_samples=3000, seed=108):
    rng = np.random.default_rng(seed)
    tensile_candidates = []

    for _ in range(n_samples):
        al = rng.uniform(5, 18, size=1)
        tm_core = rng.uniform(14, 30, size=4)
        add_mask = rng.choice([0.0, 1.0], size=3, p=[0.45, 0.55])
        adds = rng.uniform(0, 7, size=3) * add_mask

        raw = np.concatenate([al, tm_core, adds])
        at_pct = (raw / raw.sum()) * 100.0
        comp_dict = {el: round(val, 2) for el, val in zip(SCREEN_ELEMENTS, at_pct) if val > 0.1}

        vec_val = sum((comp_dict[el] / 100.0) * ELEM_PROPS[el][1] for el in comp_dict)
        if vec_val < 7.5:
            continue

        rho = sum((comp_dict[el] / 100.0) * ELEM_DENSITY[el] for el in comp_dict)
        phase = 'FCC' if vec_val >= 8.0 else 'Multi-phase'

        ys_tensile_cast = predictor.predict(comp_dict, temp_c=25.0, test_type='T', processing='CAST', phase_type=phase)
        ys_tensile_wrought = predictor.predict(comp_dict, temp_c=25.0, test_type='T', processing='WROUGHT', phase_type=phase)

        tensile_candidates.append({
            'Composition (at.%)': " ".join(f"{el}{val:.1f}" for el, val in comp_dict.items()),
            'VEC': round(vec_val, 2),
            'Est. Phase': phase,
            'Density (g/cm3)': round(rho, 2),
            'Pred. Tensile YS Cast (MPa)': round(ys_tensile_cast, 1),
            'Pred. Tensile YS Wrought (MPa)': round(ys_tensile_wrought, 1),
            'Specific Tensile YS Cast (kN*m/kg)': round(ys_tensile_cast / rho, 1)
        })

    return pd.DataFrame(tensile_candidates).sort_values(
        'Specific Tensile YS Cast (kN*m/kg)', ascending=False
    ).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict Yield Strength of Al-Cr-Co-Ni-Fe-based HEAs.')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH, help='Path to local input CSV dataset.')
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT_DIR, help='Directory for saved models, CSVs, and plots.')
    parser.add_argument('--trials', type=int, default=DEFAULT_N_TRIALS, help='Number of Optuna tuning trials.')
    parser.add_argument('--boot', type=int, default=DEFAULT_N_BOOT, help='Number of bootstrap models for uncertainty.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed.')
    parser.add_argument('--show-plots', action='store_true', help='Display interactive matplotlib windows.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_sub, x_xgb, y_xgb, comp_groups, active_elements, cat_features, num_features, h_mat = load_and_prepare_data(args.data)
    best_xgb, oof_xgb, best_params = tune_and_train_xgb(
        x_xgb, y_xgb, comp_groups, cat_features, num_features, n_trials=args.trials, seed=args.seed
    )
    print("Best Hyperparameters:", best_params)

    export_diagnostics(best_xgb, x_xgb, y_xgb, oof_xgb, df_sub, args.out_dir, show_plots=args.show_plots)

    predictor = YSPredictor(best_xgb, x_xgb.columns, cat_features, active_elements, h_mat)
    predictor.save(os.path.join(args.out_dir, 'hea_xgboost_ys_model.joblib'))

    ys_base = predictor.predict({'Al': 20, 'Co': 20, 'Cr': 20, 'Fe': 20, 'Ni': 20}, temp_c=25, test_type='C', processing='CAST', phase_type='BCC')
    ys_mo = predictor.predict({'Al': 19, 'Co': 19, 'Cr': 19, 'Fe': 19, 'Ni': 19, 'Mo': 5}, temp_c=25, test_type='C', processing='CAST', phase_type='BCC')
    print(f"\nPredicted Compressive YS (AlCoCrFeNi, Cast, 25 C)      : {ys_base:.1f} MPa")
    print(f"Predicted Compressive YS (AlCoCrFeNiMo_0.26, Cast, 25 C): {ys_mo:.1f} MPa")

    screen_df, top10 = screen_compressive_candidates(predictor, n_samples=2000, seed=args.seed)
    print("\nTop 10 Candidates by Specific Yield Strength:")
    print(top10.to_string(index=False))
    plot_screening_pareto(screen_df, top10, args.out_dir, show_plots=args.show_plots)
    screen_df.sort_values('Specific YS (kN*m/kg)', ascending=False).head(50).to_csv(
        os.path.join(args.out_dir, 'hea_screened_top50_candidates.csv'), index=False
    )

    val_df = evaluate_bootstrap_uncertainty(predictor, df_sub, x_xgb, y_xgb, comp_groups, top10, n_boot=args.boot)
    print("\nNearest-Neighbor & Bootstrap Uncertainty Check (Top 10):")
    print(val_df.to_string(index=False))

    tensile_df = screen_tensile_high_vec(predictor, n_samples=3000, seed=args.seed + 66)
    print(f"\nTop 10 Ductile Tensile Candidates (VEC >= 7.5, n={len(tensile_df)} screened):")
    print(tensile_df.head(10).to_string(index=False))
    tensile_df.head(50).to_csv(os.path.join(args.out_dir, 'hea_screened_tensile_high_vec_top50.csv'), index=False)


if __name__ == '__main__':
    main()