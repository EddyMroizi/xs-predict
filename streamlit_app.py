import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.stats import genpareto
import xgboost as xgb
import lightgbm as lgb

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XS-Predict — Tarification XS Cat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Chargement des donnees ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    X_train      = pd.read_csv("X_train.csv")
    y_freq_train = pd.read_csv("y_freq_train.csv").squeeze()
    y_sev_train  = pd.read_csv("y_sev_train.csv").squeeze()

    ratio = (y_freq_train == 0).sum() / (y_freq_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        scale_pos_weight=ratio, objective="binary:logistic",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_freq_train, verbose=False)

    mask = y_freq_train == 1
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        objective="huber", alpha=0.9, random_state=42,
        n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train[mask], y_sev_train)

    return xgb_model, lgb_model

@st.cache_data
def load_data():
    contrats     = pd.read_csv("dataset_xs_predict.csv")
    resultats_mc = pd.read_csv("resultats_mc.csv", index_col=0)
    test_df      = pd.read_csv("test_predictions.csv")
    X_test       = pd.read_csv("X_test.csv")
    shap_xgb     = pd.read_csv("shap_values_xgb.csv")
    return contrats, resultats_mc, test_df, X_test, shap_xgb

@st.cache_data
def run_monte_carlo(priorite, portee, n_simul, xi_fit, sigma_fit, lambda_annuel):
    np.random.seed(42)
    couts_annuels = np.zeros(n_simul)
    for i in range(n_simul):
        n = np.random.poisson(lambda_annuel)
        if n == 0:
            continue
        excedents        = genpareto.rvs(c=xi_fit, scale=sigma_fit, loc=0, size=n)
        charges          = np.minimum(excedents, portee)
        couts_annuels[i] = charges.sum()
    return couts_annuels

xgb_model, lgb_model = load_models()
contrats, resultats_mc, test_df, X_test, shap_xgb = load_data()

xi_fit        = float(resultats_mc.loc["xi_fit",        "valeur"])
sigma_fit     = float(resultats_mc.loc["sigma_fit",     "valeur"])
lambda_annuel = float(resultats_mc.loc["lambda_annuel", "valeur"])

contrats["prime_fictive"] = contrats["somme_assuree"] * 0.005
primes_annuelles = contrats.groupby("annee")["prime_fictive"].sum().mean()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("XS-Predict — Tarification XS Cat")
st.markdown("*Modelisation ML des sinistres extremes — Au-dela du Burning Cost*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Parametres du traite")
st.sidebar.markdown("---")

priorite = st.sidebar.slider(
    "Priorite XS (euros)",
    min_value=50000, max_value=500000,
    value=200000, step=25000
)

portee = st.sidebar.slider(
    "Portee XS (euros)",
    min_value=100000, max_value=2000000,
    value=800000, step=100000
)

scenario = st.sidebar.selectbox(
    "Scenario GPD",
    ["Central (xi=0.44)", "Optimiste (xi=0.25)", "Pessimiste (xi=0.65)"]
)

n_simul = st.sidebar.selectbox(
    "Nb simulations Monte Carlo",
    [10000, 50000, 100000],
    index=0
)

xi_scenario = {
    "Central (xi=0.44)"   : 0.44,
    "Optimiste (xi=0.25)" : 0.25,
    "Pessimiste (xi=0.65)": 0.65
}
xi_use = xi_scenario[scenario]

st.sidebar.markdown("---")
st.sidebar.markdown("**Plafond** : {:,} euros".format(priorite + portee))
st.sidebar.markdown("**GPD xi** : {}".format(xi_use))
st.sidebar.markdown("**Lambda** : {:.1f} sinistres XS/an".format(lambda_annuel))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Monte Carlo",
    "QP vs XS",
    "ML par contrat",
    "SHAP"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Simulation Monte Carlo + GPD")
    st.markdown("Traite **{:,} XS {:,}** — Scenario : **{}**".format(portee, priorite, scenario))

    with st.spinner("Simulation en cours..."):
        couts_annuels = run_monte_carlo(
            priorite, portee, n_simul, xi_use, sigma_fit, lambda_annuel
        )

    prime_pure = couts_annuels.mean()
    ecart_type = couts_annuels.std()
    var_99     = np.percentile(couts_annuels, 99)
    tvar_99    = couts_annuels[couts_annuels >= var_99].mean()
    payback    = portee / prime_pure if prime_pure > 0 else 99
    rol        = prime_pure / portee * 100
    bc_observe = float(resultats_mc.loc["charge_observee", "valeur"])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Prime pure MC", "{:.2f} M€".format(prime_pure / 1e6))
    with col2:
        st.metric("VaR 99%", "{:.2f} M€".format(var_99 / 1e6))
    with col3:
        st.metric("TVaR 99%", "{:.2f} M€".format(tvar_99 / 1e6))
    with col4:
        st.metric("Payback", "{:.1f} ans".format(payback))
    with col5:
        st.metric("ROL", "{:.1f}%".format(rol))

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(7, 4))
        couts_nonzero = couts_annuels[couts_annuels > 0]
        ax.hist(couts_nonzero / 1e6, bins=60, color="#3b82f6", alpha=0.75, edgecolor="white")
        ax.axvline(prime_pure / 1e6, color="#22c55e", lw=2,
                   label="Prime pure : {:.2f}M€".format(prime_pure / 1e6))
        ax.axvline(var_99 / 1e6, color="#f97316", lw=2, linestyle="--",
                   label="VaR 99% : {:.2f}M€".format(var_99 / 1e6))
        ax.axvline(tvar_99 / 1e6, color="#ef4444", lw=2, linestyle=":",
                   label="TVaR 99% : {:.2f}M€".format(tvar_99 / 1e6))
        ax.axvline(bc_observe / 1e6, color="#a855f7", lw=2, linestyle="-.",
                   label="BC observe : {:.2f}M€".format(bc_observe / 1e6))
        ax.set_xlabel("Cout reassureur annuel (M€)")
        ax.set_ylabel("Nb simulations")
        ax.set_title("Distribution du cout reassureur")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        percentiles = np.arange(50, 100, 0.5)
        seuils      = np.percentile(couts_annuels, percentiles)
        periodes    = 1 / (1 - percentiles / 100)
        ax2.semilogy(seuils / 1e6, periodes, color="#3b82f6", lw=2.5)
        ax2.axhline(10,  color="#22c55e", lw=1.5, linestyle="--", label="10 ans")
        ax2.axhline(100, color="#f97316", lw=1.5, linestyle="--", label="100 ans")
        ax2.axhline(200, color="#ef4444", lw=1.5, linestyle="--", label="200 ans")
        ax2.set_xlabel("Cout reassureur (M€)")
        ax2.set_ylabel("Periode de retour (ans)")
        ax2.set_title("Courbe des periodes de retour")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, which="both")
        st.pyplot(fig2)
        plt.close()

    st.markdown("### Tableau de tarification")
    df_recap = pd.DataFrame({
        "Metrique": ["Prime pure MC", "Ecart type", "Mediane",
                     "VaR 99%", "TVaR 99%", "Payback", "ROL"],
        "Valeur": [
            "{:,.0f} €".format(prime_pure),
            "{:,.0f} €".format(ecart_type),
            "{:,.0f} €".format(np.median(couts_annuels)),
            "{:,.0f} €".format(var_99),
            "{:,.0f} €".format(tvar_99),
            "{:.1f} ans".format(payback),
            "{:.2f}%".format(rol)
        ]
    })
    st.dataframe(df_recap, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QP vs XS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Comparaison QP vs XS")
    st.markdown("Meme budget reassurance — quelle structure protege le mieux le S/P net ?")

    with st.spinner("Calcul en cours..."):
        np.random.seed(42)
        charge_brute = np.zeros(n_simul)
        charge_xs_s  = np.zeros(n_simul)
        for i in range(n_simul):
            n = np.random.poisson(lambda_annuel)
            if n == 0:
                continue
            exc             = genpareto.rvs(c=xi_use, scale=sigma_fit, loc=0, size=n)
            charge_xs_s[i]  = np.minimum(exc, portee).sum()
            charge_brute[i] = (priorite + exc).sum()

        charge_nette_xs = charge_brute - charge_xs_s
        sp_net_xs       = charge_nette_xs / primes_annuelles
        prime_xs        = charge_xs_s.mean()
        taux_eq         = min(prime_xs / primes_annuelles, 0.99)
        charge_nette_qp = (1 - taux_eq) * charge_brute
        sp_net_qp       = charge_nette_qp / (primes_annuelles - taux_eq * primes_annuelles)

    var99_qp  = np.percentile(sp_net_qp, 99)
    var99_xs  = np.percentile(sp_net_xs, 99)
    tvar99_qp = sp_net_qp[sp_net_qp >= var99_qp].mean()
    tvar99_xs = sp_net_xs[sp_net_xs >= var99_xs].mean()
    gagnant   = "XS" if sp_net_xs.var() < sp_net_qp.var() else "QP"

    st.success("Structure optimale : **{}** (variance S/P minimale)".format(gagnant))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("S/P net moyen XS", "{:.1f}%".format(sp_net_xs.mean() * 100))
    with col2:
        st.metric("Variance S/P XS", "{:.3f}".format(sp_net_xs.var()))
    with col3:
        st.metric("VaR 99% S/P XS", "{:.1f}%".format(var99_xs * 100))
    with col4:
        st.metric("Taux cession QP equiv.", "{:.1f}%".format(taux_eq * 100))

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sp_net_qp * 100, bins=60, alpha=0.6, color="#3b82f6",
                density=True, label="QP {:.0f}% (var={:.3f})".format(taux_eq * 100, sp_net_qp.var()))
        ax.hist(sp_net_xs * 100, bins=60, alpha=0.6, color="#ef4444",
                density=True, label="XS (var={:.3f})".format(sp_net_xs.var()))
        ax.axvline(sp_net_qp.mean() * 100, color="#3b82f6", lw=2, linestyle="--")
        ax.axvline(sp_net_xs.mean() * 100, color="#ef4444", lw=2, linestyle="--")
        ax.set_xlabel("S/P net (%)")
        ax.set_ylabel("Densite")
        ax.set_title("Distribution S/P net QP vs XS")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        metriques = ["S/P moyen", "Ecart type", "VaR 99%", "TVaR 99%"]
        vals_qp   = [sp_net_qp.mean() * 100, sp_net_qp.std() * 100,
                     var99_qp * 100, tvar99_qp * 100]
        vals_xs   = [sp_net_xs.mean() * 100, sp_net_xs.std() * 100,
                     var99_xs * 100, tvar99_xs * 100]
        x = np.arange(len(metriques))
        w = 0.35
        ax2.bar(x - w / 2, vals_qp, w, color="#3b82f6", alpha=0.8,
                label="QP", edgecolor="white")
        ax2.bar(x + w / 2, vals_xs, w, color="#ef4444", alpha=0.8,
                label="XS", edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metriques, fontsize=9)
        ax2.set_ylabel("Valeur (%)")
        ax2.set_title("Comparaison QP vs XS")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, axis="y")
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PAR CONTRAT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Prediction ML par contrat")
    st.markdown("XGBoost predit la probabilite de declenchement XS — LightGBM predit la charge.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nb contrats test 2020", "{:,}".format(len(test_df)))
    with col2:
        st.metric("Taux declenchement moyen",
                  "{:.1f}%".format(test_df["proba_declenchement"].mean() * 100))
    with col3:
        st.metric("Prime pure ML moy. / contrat",
                  "{:,.0f} €".format(test_df["prime_pure_ml"].mean()))

    st.markdown("---")
    st.markdown("### Top 20 contrats les plus dangereux")

    top20 = test_df.nlargest(20, "prime_pure_ml")[
        ["contrat_id", "branche", "zone_geo", "secteur",
         "somme_assuree", "proba_declenchement", "charge_esperee_ml", "prime_pure_ml"]
    ].copy()
    top20["somme_assuree"]       = top20["somme_assuree"].apply(lambda x: "{:,.0f} €".format(x))
    top20["proba_declenchement"] = top20["proba_declenchement"].apply(lambda x: "{:.1f}%".format(x * 100))
    top20["charge_esperee_ml"]   = top20["charge_esperee_ml"].apply(lambda x: "{:,.0f} €".format(x))
    top20["prime_pure_ml"]       = top20["prime_pure_ml"].apply(lambda x: "{:,.0f} €".format(x))
    st.dataframe(top20, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(test_df["proba_declenchement"], bins=40,
                color="#3b82f6", alpha=0.75, edgecolor="white")
        ax.axvline(test_df["proba_declenchement"].mean(), color="#ef4444", lw=2,
                   linestyle="--",
                   label="Moyenne : {:.1f}%".format(
                       test_df["proba_declenchement"].mean() * 100))
        ax.set_xlabel("Probabilite de declenchement XS")
        ax.set_ylabel("Nb contrats")
        ax.set_title("Distribution des probabilites XGBoost")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        taux_zone = test_df.groupby("zone_geo")["proba_declenchement"].mean().sort_values(
            ascending=False)
        colors = ["#ef4444", "#f97316", "#22c55e"][:len(taux_zone)]
        bars = ax2.bar(taux_zone.index, taux_zone.values * 100,
                       color=colors, edgecolor="white")
        for bar, val in zip(bars, taux_zone.values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     "{:.1f}%".format(val * 100),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Proba declenchement moyenne (%)")
        ax2.set_title("Proba declenchement par zone geo")
        ax2.grid(alpha=0.3, axis="y")
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Explicabilite SHAP")
    st.markdown("Pourquoi le modele predit ce risque pour ce contrat ?")

    shap_cols = [c.replace("shap_", "") for c in shap_xgb.columns]
    mean_shap = shap_xgb.abs().mean()
    mean_shap.index = shap_cols
    mean_shap = mean_shap.sort_values(ascending=True).tail(10)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(7, 5))
        mean_shap.plot(kind="barh", ax=ax, color="#3b82f6", alpha=0.8)
        ax.set_title("Feature Importance SHAP XGBoost")
        ax.set_xlabel("|SHAP value| moyen")
        ax.grid(alpha=0.3, axis="x")
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.markdown("### Analyser un contrat specifique")
        contrat_ids = test_df["contrat_id"].astype(int).tolist()
        idx_max     = test_df["prime_pure_ml"].idxmax()
        contrat_sel = st.selectbox("Choisir un contrat", contrat_ids, index=idx_max)

        row = test_df[test_df["contrat_id"] == contrat_sel].iloc[0]
        st.markdown("**Branche** : {}".format(row["branche"]))
        st.markdown("**Zone geo** : {}".format(row["zone_geo"]))
        st.markdown("**Secteur** : {}".format(row["secteur"]))
        st.markdown("**Somme assuree** : {:,.0f} €".format(row["somme_assuree"]))
        st.markdown("**Proba declenchement** : {:.1f}%".format(
            row["proba_declenchement"] * 100))
        st.markdown("**Prime pure ML** : {:,.0f} €".format(row["prime_pure_ml"]))

    st.markdown("---")
    st.markdown("### Waterfall SHAP — Contrat {}".format(contrat_sel))

    idx_sel    = test_df[test_df["contrat_id"] == contrat_sel].index[0]
    shap_row   = shap_xgb.iloc[idx_sel].values
    feat_names = shap_cols
    feat_vals  = X_test.iloc[idx_sel].values

    sorted_idx = np.argsort(np.abs(shap_row))[-10:]
    shap_top   = shap_row[sorted_idx]
    feat_top   = [feat_names[i] for i in sorted_idx]
    vals_top   = feat_vals[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#ef4444" if v > 0 else "#3b82f6" for v in shap_top]
    ax.barh(range(len(feat_top)), shap_top, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(feat_top)))
    ax.set_yticklabels(
        ["{} = {:.2g}".format(f, v) for f, v in zip(feat_top, vals_top)],
        fontsize=9
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Contribution SHAP")
    ax.set_title("Waterfall SHAP — Contrat {} ({} | {})".format(
        contrat_sel, row["branche"], row["zone_geo"]))
    ax.grid(alpha=0.3, axis="x")
    st.pyplot(fig)
    plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*XS-Predict*")