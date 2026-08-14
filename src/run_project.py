"""Proyecto final ESAN - Grupo 4.

Reproduce el generador sintético, la comparación de modelos y el benchmark
Optimización Bayesiana vs Random Search. Las hipótesis del generador original
se mantienen intactas.
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_synthetic_data(seed: int = SEED) -> pd.DataFrame:
    """Genera exactamente las mismas hipótesis sintéticas del notebook original."""
    rng = np.random.default_rng(seed)
    n = 50_000
    segmentos = ["Jovenes", "Adultos", "Familias", "Premium"]
    canales = ["WhatsApp", "Chatbot Web", "SMS", "Email"]
    wordings = ["Directo", "Digital", "Beneficio", "Urgencia"]
    beneficios = ["Sin beneficio", "Descuento 10%", "Descuento 20%", "Bono datos"]
    flujos = ["Corto", "Medio", "Largo"]
    dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]

    df = pd.DataFrame({
        "mes": rng.integers(1, 13, n),
        "segmento": rng.choice(segmentos, n, p=[0.34, 0.31, 0.23, 0.12]),
        "antiguedad_meses": np.clip(rng.normal(34, 20, n).round(), 1, 120).astype(int),
        "plan_soles": np.clip(rng.normal(79, 35, n), 29, 250).round(2),
        "interacciones_previas": np.clip(rng.poisson(3.2, n), 0, 18),
        "campanas_ultimos_30d": np.clip(rng.poisson(2.1, n), 0, 10),
        "hora": rng.integers(8, 22, n),
        "dia_semana": rng.choice(dias, n),
        "canal": rng.choice(canales, n, p=[0.44, 0.25, 0.16, 0.15]),
        "wording": rng.choice(wordings, n),
        "beneficio": rng.choice(beneficios, n, p=[0.30, 0.28, 0.22, 0.20]),
        "flujo": rng.choice(flujos, n, p=[0.42, 0.36, 0.22]),
        "pasos_flujo": 0,
        "tiempo_respuesta_seg": np.clip(rng.lognormal(mean=3.4, sigma=0.5, size=n), 5, 240).round(1),
        "dispositivo": rng.choice(["Android", "iOS", "Desktop"], n, p=[0.58, 0.28, 0.14]),
        "region": rng.choice(["Lima", "Norte", "Centro", "Sur", "Oriente"], n, p=[0.48, 0.18, 0.12, 0.14, 0.08]),
        "cliente_digital": rng.choice([0, 1], n, p=[0.28, 0.72]),
        "visitas_app_30d": np.clip(rng.poisson(5.0, n), 0, 30),
        "reclamos_90d": np.clip(rng.poisson(0.55, n), 0, 5),
        "saldo_promedio": np.clip(rng.normal(24, 18, n), 0, 120).round(2),
    })
    df["pasos_flujo"] = np.select(
        [df["flujo"].eq("Corto"), df["flujo"].eq("Medio"), df["flujo"].eq("Largo")],
        [rng.integers(2, 4, n), rng.integers(4, 6, n), rng.integers(6, 9, n)],
    )

    z = np.full(n, -2.15)
    z += df["segmento"].map({"Jovenes": 0.22, "Adultos": 0.06, "Familias": 0.02, "Premium": 0.15}).values
    z += np.where(df["hora"].between(18, 20), 0.42, 0)
    z += np.where(df["hora"].between(12, 14), 0.12, 0)
    z += np.where(df["hora"] <= 9, -0.18, 0)
    z += df["canal"].map({"WhatsApp": 0.18, "Chatbot Web": 0.25, "SMS": -0.08, "Email": -0.05}).values
    z += df["wording"].map({"Directo": 0.04, "Digital": 0.18, "Beneficio": 0.14, "Urgencia": -0.03}).values
    z += df["beneficio"].map({"Sin beneficio": -0.18, "Descuento 10%": 0.14, "Descuento 20%": 0.35, "Bono datos": 0.18}).values
    z += df["flujo"].map({"Corto": 0.28, "Medio": 0.05, "Largo": -0.28}).values
    z += 0.10 * df["cliente_digital"].values
    z += 0.010 * np.minimum(df["visitas_app_30d"].values, 12)
    z += 0.0025 * np.minimum(df["antiguedad_meses"].values, 60)
    z += 0.0015 * (df["plan_soles"].values - 70)
    z -= 0.12 * df["campanas_ultimos_30d"].values
    z -= 0.10 * df["reclamos_90d"].values
    z -= 0.015 * np.maximum(df["tiempo_respuesta_seg"].values - 30, 0) / 10
    z += np.where(
        df["segmento"].eq("Jovenes")
        & df["hora"].between(18, 20)
        & df["beneficio"].eq("Descuento 20%")
        & df["flujo"].eq("Corto"),
        0.40,
        0,
    )
    z += 0.018 * (df["mes"].values - 6)

    prob_conv = 1 / (1 + np.exp(-z))
    df["conversion_exitosa"] = rng.binomial(1, prob_conv)
    df["ctr"] = np.clip(0.10 + 0.55 * prob_conv + rng.normal(0, 0.08, n), 0, 1)
    df["contencion"] = np.clip(0.18 + 0.40 * df["flujo"].eq("Corto").astype(float) + rng.normal(0, 0.12, n), 0, 1)
    df["csat"] = np.clip(3.2 + 0.7 * df["conversion_exitosa"] - 0.18 * df["reclamos_90d"] + rng.normal(0, 0.65, n), 1, 5).round(1)
    return df


def evaluate_probs(y_true: pd.Series, probas: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probas >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, probas),
        "PR-AUC": average_precision_score(y_true, probas),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Brier": brier_score_loss(y_true, probas),
        "Threshold": threshold,
    }


def best_f1_threshold(y_true: pd.Series, probas: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.80, 151)
    return max(candidates, key=lambda t: f1_score(y_true, (probas >= t).astype(int)))


def train_and_select(df: pd.DataFrame):
    target = "conversion_exitosa"
    features = [
        "segmento", "antiguedad_meses", "plan_soles", "interacciones_previas",
        "campanas_ultimos_30d", "hora", "dia_semana", "canal", "wording",
        "beneficio", "flujo", "pasos_flujo", "tiempo_respuesta_seg",
        "dispositivo", "region", "cliente_digital", "visitas_app_30d",
        "reclamos_90d", "saldo_promedio",
    ]
    train = df[df["mes"].between(1, 8)].copy()
    valid = df[df["mes"].between(9, 10)].copy()
    test = df[df["mes"].between(11, 12)].copy()
    x_train, y_train = train[features], train[target]
    x_valid, y_valid = valid[features], valid[target]
    x_test, y_test = test[features], test[target]

    categorical_cols = x_train.select_dtypes(include="object").columns.tolist()
    if "hora" not in categorical_cols:
        categorical_cols.append("hora")
    numeric_cols = [c for c in features if c not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight=None, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=12, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=320, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }

    fitted, rows = {}, []
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(x_train, y_train)
        train_p = pipe.predict_proba(x_train)[:, 1]
        valid_p = pipe.predict_proba(x_valid)[:, 1]
        threshold = best_f1_threshold(y_valid, valid_p)
        rows.append({"Modelo": name, "Train AUC": roc_auc_score(y_train, train_p), **evaluate_probs(y_valid, valid_p, threshold)})
        fitted[name] = pipe

    validation_df = pd.DataFrame(rows).set_index("Modelo")
    ranking = validation_df.sort_values(by=["PR-AUC", "AUC", "Brier", "F1"], ascending=[False, False, True, False])
    winner_name = ranking.index[0]
    winner = fitted[winner_name]
    threshold = float(validation_df.loc[winner_name, "Threshold"])
    test_p = winner.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_probs(y_test, test_p, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, (test_p >= threshold).astype(int)).ravel()
    return features, test, winner_name, winner, threshold, validation_df, test_metrics, (int(tn), int(fp), int(fn), int(tp))


def build_campaign_space(df: pd.DataFrame, features: list[str], winner):
    campaign_space = pd.DataFrame(list(product(
        ["Jovenes", "Adultos", "Familias", "Premium"],
        [10, 13, 16, 19, 21],
        ["WhatsApp", "Chatbot Web", "SMS", "Email"],
        ["Directo", "Digital", "Beneficio", "Urgencia"],
        ["Sin beneficio", "Descuento 10%", "Descuento 20%", "Bono datos"],
        ["Corto", "Medio", "Largo"],
    )), columns=["segmento", "hora", "canal", "wording", "beneficio", "flujo"])

    profile = {
        "antiguedad_meses": int(df["antiguedad_meses"].median()),
        "plan_soles": float(df["plan_soles"].median()),
        "interacciones_previas": int(df["interacciones_previas"].median()),
        "campanas_ultimos_30d": 2,
        "dia_semana": "Jueves",
        "pasos_flujo": 3,
        "tiempo_respuesta_seg": float(df["tiempo_respuesta_seg"].median()),
        "dispositivo": "Android", "region": "Lima", "cliente_digital": 1,
        "visitas_app_30d": int(df["visitas_app_30d"].median()),
        "reclamos_90d": 0, "saldo_promedio": float(df["saldo_promedio"].median()),
    }
    cost_benefit = {"Sin beneficio": 0.000, "Descuento 10%": 0.020, "Descuento 20%": 0.055, "Bono datos": 0.030}
    cost_channel = {"WhatsApp": 0.010, "Chatbot Web": 0.006, "SMS": 0.015, "Email": 0.004}
    cost_flow = {"Corto": 0.000, "Medio": 0.006, "Largo": 0.015}

    rows = []
    for _, r in campaign_space.iterrows():
        row = dict(profile)
        row.update(r.to_dict())
        row["pasos_flujo"] = {"Corto": 3, "Medio": 5, "Largo": 7}[row["flujo"]]
        rows.append(row)
    scoring_x = pd.DataFrame(rows)[features]
    p = winner.predict_proba(scoring_x)[:, 1]
    penalty = campaign_space["beneficio"].map(cost_benefit).values + campaign_space["canal"].map(cost_channel).values + campaign_space["flujo"].map(cost_flow).values
    campaign_space["prob_conversion_estimada"] = p
    campaign_space["score_negocio"] = p - penalty
    return campaign_space


def run_bo(x_encoded: np.ndarray, y_true: np.ndarray, n_trials=30, n_initial=5, kappa=1.5, seed=42):
    rng = np.random.default_rng(seed)
    observed = list(rng.choice(len(y_true), size=n_initial, replace=False))
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
    history, running_best = [], -np.inf
    for idx in observed:
        running_best = max(running_best, y_true[idx])
        history.append({"iteracion": len(history) + 1, "indice": idx, "score": y_true[idx], "best_score": running_best})
    while len(observed) < n_trials:
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
        gp.fit(x_encoded[observed], y_true[observed])
        mu, std = gp.predict(x_encoded, return_std=True)
        ucb = mu + kappa * std
        ucb[observed] = -np.inf
        next_idx = int(np.argmax(ucb))
        observed.append(next_idx)
        running_best = max(running_best, y_true[next_idx])
        history.append({"iteracion": len(observed), "indice": next_idx, "score": y_true[next_idx], "best_score": running_best})
    return pd.DataFrame(history), observed


def main() -> None:
    df = generate_synthetic_data()
    features, test, winner_name, winner, threshold, validation_df, test_metrics, cm = train_and_select(df)
    campaign_space = build_campaign_space(df, features, winner)
    bo_cols = ["segmento", "hora", "canal", "wording", "beneficio", "flujo"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_space = encoder.fit_transform(campaign_space[bo_cols])
    y_obj = campaign_space["score_negocio"].values

    history, observed = run_bo(x_space, y_obj, seed=SEED)
    best_idx = int(observed[int(np.argmax(y_obj[observed]))])
    best = campaign_space.loc[best_idx]

    def random_curve(seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_obj), size=30, replace=False)
        return np.maximum.accumulate(y_obj[idx])

    reps = 20
    bo_curves = np.vstack([run_bo(x_space, y_obj, seed=100 + s)[0]["best_score"].values for s in range(reps)])
    rs_curves = np.vstack([random_curve(100 + s) for s in range(reps)])
    target_95 = 0.95 * float(y_obj.max())
    first_reach = lambda c: (int(np.where(c >= target_95)[0][0] + 1) if np.any(c >= target_95) else np.nan)
    bo_reach = np.array([first_reach(c) for c in bo_curves], dtype=float)
    rs_reach = np.array([first_reach(c) for c in rs_curves], dtype=float)

    benchmark = pd.DataFrame({
        "Método": ["Optimización Bayesiana", "Random Search"],
        "Mejor score medio al final": [bo_curves[:, -1].mean(), rs_curves[:, -1].mean()],
        "Desv. estándar score final": [bo_curves[:, -1].std(), rs_curves[:, -1].std()],
        "Tasa éxito >=95% óptimo": [np.isfinite(bo_reach).mean(), np.isfinite(rs_reach).mean()],
        "Iteración media si alcanza 95%": [np.nanmean(bo_reach), np.nanmean(rs_reach)],
    })

    validation_df.to_csv(OUTPUT_DIR / "metricas_modelos_validation.csv")
    pd.DataFrame([{"Modelo": winner_name, **test_metrics}]).to_csv(OUTPUT_DIR / "metricas_ganador_test.csv", index=False)
    benchmark.to_csv(OUTPUT_DIR / "benchmark_bo_vs_random.csv", index=False)
    campaign_space.loc[[best_idx]].to_csv(OUTPUT_DIR / "recomendacion_bo.csv", index=False)

    summary = {
        "n_interacciones": int(len(df)), "n_columnas_totales": int(len(df.columns)),
        "n_features_modelo": len(features), "conversion_media": float(df["conversion_exitosa"].mean()),
        "modelo_ganador": winner_name, "threshold": threshold, "test_metrics": test_metrics,
        "confusion_matrix": dict(zip(["TN", "FP", "FN", "TP"], cm)),
        "bo_recomendacion": {c: (int(best[c]) if c == "hora" else str(best[c])) for c in bo_cols},
        "bo_prob_conversion": float(best["prob_conversion_estimada"]), "bo_score": float(best["score_negocio"]),
        "bo_trial_mejor": int(history.loc[history["best_score"].idxmax(), "iteracion"]),
        "bo_final_mean": float(bo_curves[:, -1].mean()), "rs_final_mean": float(rs_curves[:, -1].mean()),
        "bo_success_95": float(np.isfinite(bo_reach).mean()), "rs_success_95": float(np.isfinite(rs_reach).mean()),
        "benchmark_repeticiones": reps, "presupuesto_pruebas": 30,
    }
    with open(OUTPUT_DIR / "resumen_resultados_ppt.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
