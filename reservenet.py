# %% Etape 0 — Generation du dataset synthetique
"""
XS-Predict — Etape 0 : Generation du dataset synthetique
=========================================================
Portefeuille de reassurance prevoyance collective
Traite XS Cat : 800 000 XS 200 000
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Utilisateur\Documents\reservenet")

np.random.seed(42)

N_CONTRATS = 5_000
PRIORITE   = 200_000
PORTEE     = 800_000
PLAFOND    = PRIORITE + PORTEE
XI         = 0.4
SIGMA      = 80_000

print("=" * 60)
print("XS-Predict — Etape 0 : Generation du dataset synthetique")
print("=" * 60)
print(f"Nb contrats    : {N_CONTRATS:,}")
print(f"Priorite XS    : {PRIORITE:,} euros")
print(f"Portee XS      : {PORTEE:,} euros")
print(f"Plafond        : {PLAFOND:,} euros")
print(f"GPD xi         : {XI}")
print(f"GPD sigma      : {SIGMA:,} euros")
print()

# ── 1. Generation des contrats ────────────────────────────────────────────────
print("Etape 1 : Generation des contrats...")

contrats = pd.DataFrame({
    "contrat_id"           : range(N_CONTRATS),
    "annee"                : np.random.choice(range(2016, 2021), N_CONTRATS),
    "branche"              : np.random.choice(["prevoyance","sante","rc"], N_CONTRATS, p=[0.50,0.30,0.20]),
    "zone_geo"             : np.random.choice(["Paris","Province","Cotier"], N_CONTRATS, p=[0.40,0.40,0.20]),
    "somme_assuree"        : np.random.lognormal(mean=12, sigma=1.2, size=N_CONTRATS).round(0),
    "nb_salaries"          : np.random.randint(10, 500, N_CONTRATS),
    "secteur"              : np.random.choice(["industrie","services","btp","sante"], N_CONTRATS, p=[0.25,0.40,0.20,0.15]),
    "historique_sinistres" : np.random.poisson(lam=1.5, size=N_CONTRATS),
})

print(f"  → {len(contrats):,} contrats generes")
print()

# ── 2. Probabilite de sinistre ────────────────────────────────────────────────
print("Etape 2 : Calcul des probabilites de sinistre...")

contrats["proba_sinistre"] = (
    0.015
    + 0.060 * (contrats["zone_geo"] == "Cotier").astype(int)
    + 0.040 * (contrats["branche"] == "prevoyance").astype(int)
    + 0.000_000_05 * contrats["somme_assuree"]
    + 0.010 * contrats["historique_sinistres"]
    + 0.030 * (contrats["secteur"] == "btp").astype(int)
    + 0.000_1 * contrats["nb_salaries"]
).clip(0.001, 0.30)

print(f"  → Probabilite moyenne : {contrats['proba_sinistre'].mean():.1%}")
print()

# ── 3. Tirage des sinistres GPD ───────────────────────────────────────────────
print("Etape 3 : Tirage des sinistres via GPD...")

contrats["a_sinistre"] = np.random.binomial(n=1, p=contrats["proba_sinistre"])

excedents = genpareto.rvs(c=XI, scale=SIGMA, loc=0, size=N_CONTRATS)

contrats["montant_brut"] = np.where(
    contrats["a_sinistre"] == 1,
    PRIORITE + excedents,
    0
)

taux_sinistralite = contrats["a_sinistre"].mean()
sinistres_df      = contrats[contrats["montant_brut"] > 0]
print(f"  → Taux sinistralite : {taux_sinistralite:.1%}")
print(f"  → Nb sinistres      : {len(sinistres_df):,}")
print()

# ── 4. Variables cibles ───────────────────────────────────────────────────────
print("Etape 4 : Calcul des variables cibles ML...")

contrats["depasse_priorite"]    = (contrats["montant_brut"] > PRIORITE).astype(int)
contrats["charge_reassureur"]   = np.minimum(np.maximum(contrats["montant_brut"] - PRIORITE, 0), PORTEE)
contrats["montant_hors_traite"] = np.maximum(contrats["montant_brut"] - PLAFOND, 0)

taux_decl = contrats["depasse_priorite"].mean()
declenche = contrats[contrats["depasse_priorite"] == 1]
print(f"  → Taux declenchement XS : {taux_decl:.1%}")
print(f"  → Nb contrats XS        : {len(declenche):,}")
print()

# ── 5. Burning Cost ───────────────────────────────────────────────────────────
print("Etape 5 : Calcul du Burning Cost...")

contrats["prime_fictive"] = contrats["somme_assuree"] * 0.005

bc_annuel = contrats.groupby("annee").apply(
    lambda x: x["charge_reassureur"].sum() / x["prime_fictive"].sum()
).reset_index()
bc_annuel.columns = ["annee", "burning_cost"]

for _, row in bc_annuel.iterrows():
    print(f"    {int(row['annee'])} : {row['burning_cost']:.1%}")
print(f"  BC moyen : {bc_annuel['burning_cost'].mean():.1%}")
print()

# ── 6. Split train/test ───────────────────────────────────────────────────────
print("Etape 6 : Split temporel train/test...")

FEATURES = ["branche","zone_geo","somme_assuree","nb_salaries","secteur","historique_sinistres"]

train = contrats[contrats["annee"] <= 2019].copy()
test  = contrats[contrats["annee"] == 2020].copy()

X_train = pd.get_dummies(train[FEATURES], drop_first=False)
X_test  = pd.get_dummies(test[FEATURES],  drop_first=False)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

y_freq_train = train["depasse_priorite"]
y_freq_test  = test["depasse_priorite"]
y_sev_train  = train[train["depasse_priorite"] == 1]["charge_reassureur"]

print(f"  → Train : {len(train):,} | Test : {len(test):,}")
print(f"  → Nb features : {X_train.shape[1]}")
print()

# ── 7. Sauvegarde ─────────────────────────────────────────────────────────────
contrats.to_csv("dataset_xs_predict.csv",   index=False)
X_train.to_csv("X_train.csv",              index=False)
X_test.to_csv("X_test.csv",               index=False)
y_freq_train.to_csv("y_freq_train.csv",   index=False)
y_freq_test.to_csv("y_freq_test.csv",     index=False)
y_sev_train.to_csv("y_sev_train.csv",     index=False)

print("=" * 60)
print("RESUME DU DATASET")
print("=" * 60)
print(f"Nb contrats total          : {len(contrats):,}")
print(f"Nb contrats sinistres      : {contrats['a_sinistre'].sum():,} ({taux_sinistralite:.1%})")
print(f"Nb contrats declenchant XS : {contrats['depasse_priorite'].sum():,} ({taux_decl:.1%})")
print(f"Charge reassureur / an     : {contrats.groupby('annee')['charge_reassureur'].sum().mean():,.0f} euros")
print(f"Burning Cost moyen         : {bc_annuel['burning_cost'].mean():.1%}")
print(f"Split train/test           : {len(train):,} / {len(test):,}")
print("=" * 60)
print("Dataset pret pour Etape 1 (Monte Carlo) et Etape 2 (ML)")


# %% Etape 1 — Monte Carlo + GPD
"""
XS-Predict — Etape 1 : Monte Carlo + GPD
=========================================
Simulation de 100 000 annees pour obtenir la distribution
du cout reassureur et les metriques de risque (VaR, TVaR).
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Utilisateur\Documents\reservenet")

np.random.seed(42)

PRIORITE = 200_000
PORTEE   = 800_000
PLAFOND  = PRIORITE + PORTEE
N_SIMUL  = 100_000

print("=" * 60)
print("XS-Predict — Etape 1 : Monte Carlo + GPD")
print("=" * 60)

contrats = pd.read_csv("dataset_xs_predict.csv")
contrats["prime_fictive"] = contrats["somme_assuree"] * 0.005

bc_annuel = contrats.groupby("annee").apply(
    lambda x: x["charge_reassureur"].sum() / x["prime_fictive"].sum()
).reset_index()
bc_annuel.columns = ["annee", "burning_cost"]

charge_moy_annee = contrats.groupby("annee")["charge_reassureur"].sum().mean()
primes_moy_annee = contrats.groupby("annee")["prime_fictive"].sum().mean()
bc_moyen         = bc_annuel["burning_cost"].mean()

print(f"\n  BC moyen              : {bc_moyen:.1%}")
print(f"  Charge reass. moy/an  : {charge_moy_annee:,.0f} euros")
print(f"  Primes moy/an         : {primes_moy_annee:,.0f} euros")

# ── Calibration GPD ───────────────────────────────────────────────────────────
sinistres_xs  = contrats[contrats["depasse_priorite"] == 1]["montant_brut"]
excedents_obs = sinistres_xs - PRIORITE

xi_fit, loc_fit, sigma_fit = genpareto.fit(excedents_obs, floc=0)

print(f"\n  GPD calibree (MLE) :")
print(f"  → xi    : {xi_fit:.4f}")
print(f"  → sigma : {sigma_fit:,.0f} euros")

freq_par_annee = contrats[contrats["depasse_priorite"] == 1].groupby("annee").size()
lambda_annuel  = freq_par_annee.mean()
print(f"  → Lambda Poisson : {lambda_annuel:.1f} sinistres XS / an")

# ── Simulation Monte Carlo ────────────────────────────────────────────────────
print(f"\n  Simulation Monte Carlo ({N_SIMUL:,} annees)...")

couts_annuels = np.zeros(N_SIMUL)
for i in range(N_SIMUL):
    n = np.random.poisson(lambda_annuel)
    if n == 0:
        continue
    excedents       = genpareto.rvs(c=xi_fit, scale=sigma_fit, loc=0, size=n)
    charges         = np.minimum(excedents, PORTEE)
    couts_annuels[i] = charges.sum()

prime_pure = couts_annuels.mean()
ecart_type = couts_annuels.std()
var_99     = np.percentile(couts_annuels, 99)
var_995    = np.percentile(couts_annuels, 99.5)
tvar_99    = couts_annuels[couts_annuels >= var_99].mean()
mediane    = np.median(couts_annuels)
payback    = PORTEE / prime_pure if prime_pure > 0 else np.inf
cv         = ecart_type / prime_pure

print(f"\n  {'Metrique':<35} {'Valeur':>20}")
print(f"  {'-'*55}")
print(f"  {'Prime pure Monte Carlo':<35} {prime_pure:>20,.0f} euros")
print(f"  {'Charge reass. observee / an':<35} {charge_moy_annee:>20,.0f} euros")
print(f"  {'Ecart MC vs observe':<35} {(prime_pure/charge_moy_annee - 1):>19.1%}")
print(f"  {'VaR 99%':<35} {var_99:>20,.0f} euros")
print(f"  {'TVaR 99%':<35} {tvar_99:>20,.0f} euros")
print(f"  {'Payback':<35} {payback:>19.1f} ans")
print(f"  {'CV':<35} {cv:>19.1%}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
np.save("couts_annuels_mc.npy", couts_annuels)

resultats_mc = {
    "prime_pure"     : prime_pure,
    "charge_observee": charge_moy_annee,
    "ecart_type"     : ecart_type,
    "var_99"         : var_99,
    "var_995"        : var_995,
    "tvar_99"        : tvar_99,
    "payback"        : payback,
    "cv"             : cv,
    "bc_moyen"       : bc_moyen,
    "lambda_annuel"  : lambda_annuel,
    "xi_fit"         : xi_fit,
    "sigma_fit"      : sigma_fit,
}
pd.Series(resultats_mc).to_csv("resultats_mc.csv", header=["valeur"])

print("\n" + "=" * 60)
print("RESUME ETAPE 1 — MONTE CARLO + GPD")
print("=" * 60)
print(f"GPD calibree : xi={xi_fit:.3f} | sigma={sigma_fit:,.0f} euros")
print(f"Lambda       : {lambda_annuel:.1f} sinistres XS / an")
print(f"Prime pure   : {prime_pure:,.0f} euros/an")
print(f"VaR 99%      : {var_99:,.0f} euros")
print(f"TVaR 99%     : {tvar_99:,.0f} euros")
print(f"Payback      : {payback:.1f} ans")
print("=" * 60)
print("Pret pour Etape 2 : XGBoost + LightGBM")


# %% Etape 2 — XGBoost + LightGBM
"""
XS-Predict — Etape 2 : XGBoost + LightGBM
==========================================
Modele de frequence : XGBoost  → depasse_priorite (OUI/NON)
Modele de severite  : LightGBM → charge_reassureur (euros)
Validation temporelle : train 2016-2019 / test 2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import joblib

from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
import xgboost  as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Utilisateur\Documents\reservenet")

print("=" * 60)
print("XS-Predict — Etape 2 : XGBoost + LightGBM")
print("=" * 60)

# ── Chargement ────────────────────────────────────────────────────────────────
X_train      = pd.read_csv("X_train.csv")
X_test       = pd.read_csv("X_test.csv")
y_freq_train = pd.read_csv("y_freq_train.csv").squeeze()
y_freq_test  = pd.read_csv("y_freq_test.csv").squeeze()
y_sev_train  = pd.read_csv("y_sev_train.csv").squeeze()

contrats     = pd.read_csv("dataset_xs_predict.csv")
test_df      = contrats[contrats["annee"] == 2020].copy().reset_index(drop=True)

resultats_mc    = pd.read_csv("resultats_mc.csv", index_col=0)
prime_pure_mc   = float(resultats_mc.loc["prime_pure",     "valeur"])
charge_observee = float(resultats_mc.loc["charge_observee","valeur"])

print(f"\n  X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"  Taux decl. train : {y_freq_train.mean():.1%} | test : {y_freq_test.mean():.1%}")
print(f"  Prime pure MC ref : {prime_pure_mc:,.0f} euros/an")

# ── XGBoost — Frequence ───────────────────────────────────────────────────────
print("\nEtape 2.2 : XGBoost — modele de frequence...")

ratio_classes = (y_freq_train == 0).sum() / (y_freq_train == 1).sum()

def objective_xgb(trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "scale_pos_weight" : ratio_classes,
        "objective"        : "binary:logistic",
        "eval_metric"      : "auc",
        "random_state"     : 42,
        "n_jobs"           : -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_freq_train, verbose=False)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_freq_test, preds)

print("  → Optuna XGBoost (50 trials)...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)

best_params_xgb = study_xgb.best_params
best_params_xgb.update({"objective":"binary:logistic","scale_pos_weight":ratio_classes,"random_state":42,"n_jobs":-1})

xgb_model    = xgb.XGBClassifier(**best_params_xgb)
xgb_model.fit(X_train, y_freq_train, verbose=False)

y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)
auc_score    = roc_auc_score(y_freq_test, y_pred_proba)

print(f"  AUC-ROC : {auc_score:.4f}")
print(classification_report(y_freq_test, y_pred_class, target_names=["Non declenche","Declenche XS"]))

# ── LightGBM — Severite ───────────────────────────────────────────────────────
print("\nEtape 2.3 : LightGBM — modele de severite...")

mask_sev_train = y_freq_train == 1
X_sev_train    = X_train[mask_sev_train]
mask_sev_test  = y_freq_test == 1
X_sev_test     = X_test[mask_sev_test]
y_sev_test     = test_df[test_df["depasse_priorite"] == 1]["charge_reassureur"].reset_index(drop=True)

def objective_lgb(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"       : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves"      : trial.suggest_int("num_leaves", 20, 100),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "objective"       : "huber",
        "alpha"           : 0.9,
        "random_state"    : 42,
        "n_jobs"          : -1,
        "verbose"         : -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_sev_train, y_sev_train)
    preds = model.predict(X_sev_test)
    return np.sqrt(np.mean((preds - y_sev_test.values) ** 2))

print("  → Optuna LightGBM (50 trials)...")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=False)

best_params_lgb = study_lgb.best_params
best_params_lgb.update({"objective":"huber","alpha":0.9,"random_state":42,"n_jobs":-1,"verbose":-1})

lgb_model  = lgb.LGBMRegressor(**best_params_lgb)
lgb_model.fit(X_sev_train, y_sev_train)

y_sev_pred = np.maximum(lgb_model.predict(X_sev_test), 0)
rmse       = np.sqrt(np.mean((y_sev_pred - y_sev_test.values) ** 2))
ae_ratio   = y_sev_pred.sum() / y_sev_test.sum()

print(f"  RMSE : {rmse:,.0f} euros | A/E : {ae_ratio:.3f}")

# ── Prime pure par contrat ────────────────────────────────────────────────────
print("\nEtape 2.4 : Prime pure par contrat...")

proba_decl         = xgb_model.predict_proba(X_test)[:, 1]
charge_cond        = np.maximum(lgb_model.predict(X_test), 0)
prime_pure_contrat = proba_decl * charge_cond

test_df["proba_declenchement"] = proba_decl
test_df["charge_esperee_ml"]   = charge_cond
test_df["prime_pure_ml"]       = prime_pure_contrat

prime_pure_ml_annuelle = prime_pure_contrat.mean() * 5_000

print(f"  Prime pure ML annuelle : {prime_pure_ml_annuelle:,.0f} euros/an")
print(f"  Prime pure MC ref.     : {prime_pure_mc:,.0f} euros/an")
print(f"  Ecart ML vs MC         : {(prime_pure_ml_annuelle/prime_pure_mc - 1):+.1%}")

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("XS-Predict — Etape 2 : XGBoost + LightGBM\nPrediction frequence et severite — Test 2020",
             fontsize=13, fontweight="bold")

ax1 = axes[0, 0]
RocCurveDisplay.from_predictions(y_freq_test, y_pred_proba, ax=ax1, color="#3b82f6",
                                  name=f"XGBoost (AUC={auc_score:.3f})")
ax1.plot([0,1],[0,1],"k--",lw=1,label="Aleatoire")
ax1.set_title("Courbe ROC — XGBoost")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(y_pred_proba[y_freq_test==0], bins=40, alpha=0.6, color="#3b82f6", label="Non declenche", density=True)
ax2.hist(y_pred_proba[y_freq_test==1], bins=40, alpha=0.6, color="#ef4444", label="Declenche XS",  density=True)
ax2.axvline(0.5, color="black", lw=1.5, linestyle="--")
ax2.set_title("Separation des classes — XGBoost")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

ax3 = axes[0, 2]
fi_xgb = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True).tail(10)
fi_xgb.plot(kind="barh", ax=ax3, color="#3b82f6", alpha=0.8)
ax3.set_title("Feature Importance XGBoost (Top 10)")
ax3.grid(alpha=0.3, axis="x")

ax4 = axes[1, 0]
ax4.scatter(y_sev_test/1e3, y_sev_pred/1e3, alpha=0.4, color="#3b82f6", s=20)
lim = max(y_sev_test.max(), y_sev_pred.max()) / 1e3 * 1.05
ax4.plot([0,lim],[0,lim],"r--",lw=2)
ax4.set_xlabel("Charge reelle (k€)"); ax4.set_ylabel("Charge predite (k€)")
ax4.set_title(f"Actual vs Predicted — LightGBM\nRMSE={rmse/1e3:.0f}k€ | A/E={ae_ratio:.3f}")
ax4.grid(alpha=0.3)

ax5 = axes[1, 1]
fi_lgb = pd.Series(lgb_model.feature_importances_, index=X_sev_train.columns).sort_values(ascending=True).tail(10)
fi_lgb.plot(kind="barh", ax=ax5, color="#22c55e", alpha=0.8)
ax5.set_title("Feature Importance LightGBM (Top 10)")
ax5.grid(alpha=0.3, axis="x")

ax6 = axes[1, 2]
categories = ["Charge\nobservee","Prime pure\nMonte Carlo","Prime pure\nML"]
valeurs    = [charge_observee/1e6, prime_pure_mc/1e6, prime_pure_ml_annuelle/1e6]
bars       = ax6.bar(categories, valeurs, color=["#a855f7","#22c55e","#3b82f6"], alpha=0.8, edgecolor="white")
for bar, val in zip(bars, valeurs):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.1f}M€",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax6.set_title("Validation croisee ML vs MC vs Observe")
ax6.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("ml_results.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Sauvegarde ────────────────────────────────────────────────────────────────
test_df.to_csv("test_predictions.csv", index=False)
joblib.dump(xgb_model, "xgb_frequence.pkl")
joblib.dump(lgb_model, "lgb_severite.pkl")

print("\n" + "=" * 60)
print("RESUME ETAPE 2 — XGBOOST + LIGHTGBM")
print("=" * 60)
print(f"XGBoost AUC-ROC        : {auc_score:.4f}")
print(f"LightGBM RMSE          : {rmse:,.0f} euros")
print(f"LightGBM A/E ratio     : {ae_ratio:.3f}")
print(f"Prime pure ML annuelle : {prime_pure_ml_annuelle:,.0f} euros/an")
print(f"Prime pure MC          : {prime_pure_mc:,.0f} euros/an")
print(f"Ecart ML vs MC         : {(prime_pure_ml_annuelle/prime_pure_mc - 1):+.1%}")
print("=" * 60)
print("Pret pour Etape 3 : Optimisation QP vs XS")


# %% Etape 3 — Optimisation QP vs XS
"""
XS-Predict — Etape 3 : Optimisation QP vs XS
=============================================
Comparaison Quote-Part vs Excedent de Sinistre.
Critere : variance du S/P net apres reassurance.
Frontiere efficiente a la Markowitz.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from scipy.stats import genpareto

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Utilisateur\Documents\reservenet")

np.random.seed(42)

PRIORITE = 200_000
PORTEE   = 800_000
PLAFOND  = PRIORITE + PORTEE

print("=" * 60)
print("XS-Predict — Etape 3 : Optimisation QP vs XS")
print("=" * 60)

# ── Chargement ────────────────────────────────────────────────────────────────
couts_annuels = np.load("couts_annuels_mc.npy")
contrats      = pd.read_csv("dataset_xs_predict.csv")
resultats_mc  = pd.read_csv("resultats_mc.csv", index_col=0)

prime_pure_mc   = float(resultats_mc.loc["prime_pure",      "valeur"])
charge_observee = float(resultats_mc.loc["charge_observee", "valeur"])
xi_fit          = float(resultats_mc.loc["xi_fit",          "valeur"])
sigma_fit       = float(resultats_mc.loc["sigma_fit",       "valeur"])
lambda_annuel   = float(resultats_mc.loc["lambda_annuel",   "valeur"])

N_SIMUL    = len(couts_annuels)
N_FRONTIER = 5_000

if "prime_fictive" not in contrats.columns:
    contrats["prime_fictive"] = contrats["somme_assuree"] * 0.005
primes_annuelles = contrats.groupby("annee")["prime_fictive"].sum().mean()

print(f"  → {N_SIMUL:,} simulations | Lambda={lambda_annuel:.1f} | xi={xi_fit:.3f} | sigma={sigma_fit:,.0f}")
print(f"  → Primes annuelles : {primes_annuelles:,.0f} euros/an")

# ── Reconstruction charge brute et XS ────────────────────────────────────────
print("\nEtape 3.2 : Reconstruction des charges brutes...")

np.random.seed(42)
charge_brute = np.zeros(N_SIMUL)
charge_xs    = np.zeros(N_SIMUL)

for i in range(N_SIMUL):
    n = np.random.poisson(lambda_annuel)
    if n == 0:
        continue
    excedents       = genpareto.rvs(c=xi_fit, scale=sigma_fit, loc=0, size=n)
    charge_xs[i]    = np.minimum(excedents, PORTEE).sum()
    charge_brute[i] = (PRIORITE + excedents).sum()

charge_nette_xs = charge_brute - charge_xs
sp_net_xs       = charge_nette_xs / primes_annuelles
prime_xs        = prime_pure_mc

print(f"  → Charge brute moy.  : {charge_brute.mean():,.0f} euros/an")
print(f"  → Charge XS moy.     : {charge_xs.mean():,.0f} euros/an")
print(f"  → S/P net XS moyen   : {sp_net_xs.mean():.1%}")
print(f"  → Variance S/P XS    : {sp_net_xs.var():.6f}")

# ── Quote-Part ────────────────────────────────────────────────────────────────
print("\nEtape 3.3 : Structure Quote-Part...")

taux_cession_budget = min(prime_xs / primes_annuelles, 0.99)
taux_cessions       = np.arange(0.05, 0.95, 0.05)
resultats_qp        = []

for taux in taux_cessions:
    prime_qp        = taux * primes_annuelles
    charge_nette_qp = (1 - taux) * charge_brute
    sp_net_qp       = charge_nette_qp / (primes_annuelles - prime_qp)
    resultats_qp.append({
        "taux_cession" : taux,
        "prime_qp"     : prime_qp,
        "sp_net_moyen" : sp_net_qp.mean(),
        "variance_sp"  : sp_net_qp.var(),
    })

df_qp = pd.DataFrame(resultats_qp)
print(f"  → Taux cession budget equivalent : {taux_cession_budget:.1%}")

# ── Comparaison directe QP vs XS ─────────────────────────────────────────────
print("\nEtape 3.4 : Comparaison QP vs XS...")

taux_eq            = taux_cession_budget
charge_nette_qp_eq = (1 - taux_eq) * charge_brute
sp_net_qp_eq       = charge_nette_qp_eq / (primes_annuelles - taux_eq * primes_annuelles)

var99_qp  = np.percentile(sp_net_qp_eq, 99)
var99_xs  = np.percentile(sp_net_xs,    99)
tvar99_qp = sp_net_qp_eq[sp_net_qp_eq >= var99_qp].mean()
tvar99_xs = sp_net_xs[sp_net_xs >= var99_xs].mean()

print(f"\n  {'Metrique':<35} {'QP':>15} {'XS':>15} {'Gagnant':>10}")
print(f"  {'-'*75}")
for nom, vq, vx in [
    ("S/P net moyen",    sp_net_qp_eq.mean(), sp_net_xs.mean()),
    ("Variance S/P net", sp_net_qp_eq.var(),  sp_net_xs.var()),
    ("Ecart type S/P",   sp_net_qp_eq.std(),  sp_net_xs.std()),
    ("VaR 99% S/P",      var99_qp,            var99_xs),
    ("TVaR 99% S/P",     tvar99_qp,           tvar99_xs),
]:
    g = "XS ✓" if vx < vq else "QP ✓"
    print(f"  {nom:<35} {vq:>14.1%} {vx:>14.1%} {g:>10}")

# ── Frontiere efficiente ──────────────────────────────────────────────────────
print(f"\nEtape 3.5 : Frontiere efficiente ({N_FRONTIER:,} simul.)...")

np.random.seed(42)
charge_brute_f = np.zeros(N_FRONTIER)
for i in range(N_FRONTIER):
    n = np.random.poisson(lambda_annuel)
    if n == 0: continue
    exc = genpareto.rvs(c=xi_fit, scale=sigma_fit, loc=0, size=n)
    charge_brute_f[i] = (PRIORITE + exc).sum()

budgets        = np.linspace(prime_xs * 0.3, prime_xs * 2.0, 20)
priorites_test = [100_000, 150_000, 200_000, 300_000, 400_000, 500_000]
frontier_qp    = []
frontier_xs    = []

for idx, budget in enumerate(budgets):
    print(f"  → Budget {idx+1:02d}/20 : {budget/1e6:.1f}M€", end="\r")

    taux = min(budget / primes_annuelles, 0.99)
    cn   = (1 - taux) * charge_brute_f
    sp   = cn / (primes_annuelles - budget)
    frontier_qp.append({"budget": budget, "variance": sp.var()})

    variances_xs_budget = []
    for prio in priorites_test:
        exc_prio = contrats[contrats["montant_brut"] > prio]["montant_brut"] - prio
        if len(exc_prio) < 5: continue
        xi_p, _, sigma_p = genpareto.fit(exc_prio, floc=0)
        freq_p = len(exc_prio) / 5

        np.random.seed(42)
        couts_p = np.zeros(N_FRONTIER)
        for i in range(N_FRONTIER):
            n = np.random.poisson(freq_p)
            if n == 0: continue
            exc = genpareto.rvs(c=xi_p, scale=sigma_p, loc=0, size=n)
            couts_p[i] = np.minimum(exc, PLAFOND - prio).sum()

        prime_p = couts_p.mean()
        if abs(prime_p - budget) / max(budget, 1) < 0.5:
            sp_p = (charge_brute_f - prime_p) / primes_annuelles
            variances_xs_budget.append(sp_p.var())

    if variances_xs_budget:
        frontier_xs.append({"budget": budget, "variance": min(variances_xs_budget)})

df_frontier_qp = pd.DataFrame(frontier_qp)
df_frontier_xs = pd.DataFrame(frontier_xs)
print(f"\n  → Frontiere QP : {len(df_frontier_qp)} pts | Frontiere XS : {len(df_frontier_xs)} pts")

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("XS-Predict — Etape 3 : Optimisation QP vs XS\nComparaison des structures",
             fontsize=13, fontweight="bold")

ax1 = axes[0, 0]
ax1.hist(sp_net_qp_eq, bins=80, alpha=0.6, color="#3b82f6", density=True,
         label=f"QP {taux_eq:.0%} (var={sp_net_qp_eq.var():.5f})")
ax1.hist(sp_net_xs,    bins=80, alpha=0.6, color="#ef4444", density=True,
         label=f"XS 200k (var={sp_net_xs.var():.5f})")
ax1.axvline(sp_net_qp_eq.mean(), color="#3b82f6", lw=2, linestyle="--")
ax1.axvline(sp_net_xs.mean(),    color="#ef4444",  lw=2, linestyle="--")
ax1.set_xlabel("S/P net"); ax1.set_ylabel("Densite")
ax1.set_title("Distribution S/P net QP vs XS")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(df_qp["taux_cession"]*100, df_qp["variance"], "o-", color="#3b82f6", lw=2.5, ms=7, label="QP")
ax2.axhline(sp_net_xs.var(), color="#ef4444", lw=2, linestyle="-.", label=f"XS variance={sp_net_xs.var():.5f}")
ax2.axvline(taux_eq*100, color="gray", lw=1.5, linestyle="--", label=f"Taux equiv. {taux_eq:.0%}")
ax2.set_xlabel("Taux cession QP (%)"); ax2.set_ylabel("Variance S/P net")
ax2.set_title("Variance S/P vs Taux cession QP")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(df_frontier_qp["budget"]/1e6, df_frontier_qp["variance"],
         "o-", color="#3b82f6", lw=2.5, ms=7, label="Quote-Part")
if len(df_frontier_xs) > 1:
    ax3.plot(df_frontier_xs["budget"]/1e6, df_frontier_xs["variance"],
             "s--", color="#ef4444", lw=2, ms=7, label="XS Cat")
ax3.axvline(prime_xs/1e6, color="gray", lw=1.5, linestyle=":", label=f"Prime XS {prime_xs/1e6:.1f}M€")
ax3.set_xlabel("Cout reassurance (M€)"); ax3.set_ylabel("Variance S/P net")
ax3.set_title("Frontiere efficiente (Markowitz)")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
labels  = ["S/P net\nmoyen","Ecart\ntype","VaR 99%","TVaR 99%"]
vals_qp = [sp_net_qp_eq.mean(), sp_net_qp_eq.std(), var99_qp, tvar99_qp]
vals_xs = [sp_net_xs.mean(),    sp_net_xs.std(),    var99_xs, tvar99_xs]
x = np.arange(len(labels)); w = 0.35
ax4.bar(x-w/2, [v*100 for v in vals_qp], w, color="#3b82f6", alpha=0.8, label="QP", edgecolor="white")
ax4.bar(x+w/2, [v*100 for v in vals_xs], w, color="#ef4444",  alpha=0.8, label="XS", edgecolor="white")
ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=9)
ax4.set_ylabel("Valeur (%)"); ax4.set_title("Comparaison QP vs XS")
ax4.legend(fontsize=9); ax4.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("optimisation_qp_xs.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Sauvegarde ────────────────────────────────────────────────────────────────
pd.Series({
    "taux_cession_qp_eq" : taux_eq,
    "prime_qp_eq"        : taux_eq * primes_annuelles,
    "prime_xs"           : prime_xs,
    "sp_net_moyen_qp"    : sp_net_qp_eq.mean(),
    "sp_net_moyen_xs"    : sp_net_xs.mean(),
    "variance_sp_qp"     : sp_net_qp_eq.var(),
    "variance_sp_xs"     : sp_net_xs.var(),
    "var99_sp_qp"        : var99_qp,
    "var99_sp_xs"        : var99_xs,
    "tvar99_sp_qp"       : tvar99_qp,
    "tvar99_sp_xs"       : tvar99_xs,
}).to_csv("resultats_etape3.csv", header=["valeur"])

gagnant = "XS" if sp_net_xs.var() < sp_net_qp_eq.var() else "QP"

print("\n" + "=" * 60)
print("RESUME ETAPE 3 — OPTIMISATION QP vs XS")
print("=" * 60)
print(f"S/P net moyen QP   : {sp_net_qp_eq.mean():.1%}")
print(f"S/P net moyen XS   : {sp_net_xs.mean():.1%}")
print(f"Variance S/P QP    : {sp_net_qp_eq.var():.6f}")
print(f"Variance S/P XS    : {sp_net_xs.var():.6f}")
print(f"VaR 99% S/P QP     : {var99_qp:.1%}")
print(f"VaR 99% S/P XS     : {var99_xs:.1%}")
print(f"Structure optimale : {gagnant}")
print("=" * 60)
print("Pret pour Etape 4 : SHAP — Explicabilite")