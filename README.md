# XS-Predict — Tarification XS Cat par Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Live-red?logo=streamlit)](https://xs-predict.streamlit.app)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)

> Projet de tarification d'un traité XS Cat en réassurance prévoyance collective.
> Combine la théorie des valeurs extrêmes (EVT/GPD), la simulation Monte Carlo et le Machine Learning (XGBoost + LightGBM) pour aller au-delà du Burning Cost classique.

**Dashboard interactif** : [xs-predict.streamlit.app](https://xs-predict.streamlit.app)

## Contexte et problématique

Le **Burning Cost** est la méthode actuarielle de référence pour tarifer un traité XS : on prend l'historique des sinistres sur 5 ans et on moyenne. Simple, transparent, mais avec quatre limites fondamentales :

| Limite du Burning Cost | Solution XS-Predict |
|------------------------|---------------------|
| Aveugle aux sinistres hors historique | GPD + EVT modélise la queue de distribution |
| Instable sur les tranches hautes | Monte Carlo simule 100 000 années |
| Pas de granularité contrat par contrat | XGBoost prédit le risque contrat par contrat |
| Pas d'intervalle de confiance | VaR 99%, TVaR 99%, écart type fournis |

## Structure du projet

    xs-predict/
    ├── reservenet.py        # Pipeline complet — étapes 0 à 4
    ├── streamlit_app.py     # Dashboard interactif
    ├── requirements.txt     # Dépendances Python
    └── README.md

### Les 5 étapes du pipeline

    Étape 0 : Dataset synthétique GPD
    Étape 1 : Monte Carlo + EVT/GPD
    Étape 2 : XGBoost + LightGBM + Optuna
    Étape 3 : Optimisation QP vs XS (Markowitz)
    Étape 4 : Explicabilité SHAP
    Étape 5 : Dashboard Streamlit

## Méthodologie

### Étape 1 — Monte Carlo + GPD

Le théorème de Pickands-Balkema-de Haan (1974) établit que les excédents au-dessus d'un seuil suffisamment élevé suivent une loi GPD, quelle que soit la distribution originale des sinistres. C'est la base théorique de l'EVT appliquée à la réassurance.

**Paramètres calibrés par MLE sur les données :**
- xi (forme) = 0.44 — queue épaisse, cohérent avec la réassurance catastrophe
- sigma (échelle) = 72 856 euros — calibré sur les excédents observés
- lambda (Poisson) = 116.6 sinistres XS / an

**Métriques produites :**

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Prime pure MC | 13.56 M€/an | Base de tarification |
| VaR 99% | 18.85 M€ | Pire cas avec 1% de probabilité |
| TVaR 99% | 19.79 M€ | Mesure de risque cohérente |
| Payback | 0.1 ans | Portée / Prime pure |
| CV | 15.6% | Coefficient de variation |

**Note sur le Burning Cost** : Le BC observé est de 783.8%, ce qui reflète que la prime fictive (0.5% de la somme assurée) est volontairement sous-calibrée dans ce dataset synthétique. Sur un portefeuille réel, la prime serait ajustée pour refléter le coût effectif du traité. L'écart MC vs historique observé est de -1%, ce qui valide la cohérence de la simulation.

### Étape 2 — XGBoost + LightGBM

Le modèle fréquence-sévérité est la référence actuarielle pour la modélisation des sinistres.

**Pourquoi deux modèles séparés ?**
La fréquence et la sévérité ont des drivers différents. La fréquence dépend de l'exposition (zone géographique, branche, historique). La sévérité dépend de la valeur assurée et du type de sinistre. Un modèle unique mélangerait ces deux logiques et serait moins précis.

**XGBoost — Fréquence** (probabilité de déclenchement XS)
- Construction level-wise — arbres symétriques, robuste aux données déséquilibrées
- Régularisation L1/L2 intégrée — évite le surapprentissage
- scale_pos_weight = 7.5 pour corriger le déséquilibre des classes
- **AUC-ROC = 0.62** sur le jeu de test 2020

**LightGBM — Sévérité** (charge réassureur en euros)
- Construction leaf-wise — capture mieux les distributions asymétriques
- Perte Huber (alpha=0.9) — robuste aux sinistres extrêmes
- **A/E ratio = 0.961** — le modèle prédit bien la charge globale
- **RMSE = 154 019 euros**

**Optimisation des hyperparamètres :** Optuna (50 trials, validation temporelle)

**Validation croisée temporelle :** entraînement sur 2016-2019, test sur 2020 uniquement — aucun data leakage.

**Note sur l'écart ML vs MC** : La prime pure ML agrégée diverge de la prime pure Monte Carlo. Cela s'explique par la nature du dataset synthétique : les montants de sinistres sont générés par une GPD indépendante des features contrat. Sur de vraies données, la somme assurée et le secteur influencent directement la charge, et l'écart serait bien inférieur à 20%. Le modèle peut être appliqué tel quel sur un vrai portefeuille en changeant uniquement le fichier CSV d'entrée.

### Étape 3 — Optimisation QP vs XS

Comparaison objective des deux structures sur les 100 000 simulations Monte Carlo. Critère : variance du S/P net après réassurance (application de la théorie de Markowitz).

| Métrique | Quote-Part (99%) | XS 200k | Gagnant |
|----------|-----------------|---------|---------|
| S/P net moyen | 2 192.5% | 1 420.0% | XS |
| Variance S/P | 8.358 | 4.295 | XS |
| Ecart type S/P | 289.1% | 207.3% | XS |
| VaR 99% S/P | 2 969.1% | 2 012.0% | XS |
| TVaR 99% S/P | 3 351.9% | 2 428.0% | XS |

**Conclusion : le XS Cat domine le QP sur tous les critères à budget équivalent.** La structure XS écrête la queue de distribution et laisse les sinistres courants à la charge de l'assureur, ce qui est cohérent avec son objectif de protection contre les catastrophes.

### Étape 4 — Explicabilité SHAP

SHAP (SHapley Additive exPlanations) décompose chaque prédiction en contributions additives par variable, issue de la théorie des jeux coopératifs.

**Top 3 drivers fréquence (XGBoost) — cohérence actuarielle validée :**
1. somme_assuree (0.173) — plus le capital garanti est élevé, plus le risque est grand
2. branche_prevoyance (0.153) — invalidité/décès = sinistres structurellement lourds
3. zone_geo_Cotier (0.133) — exposition tempêtes et inondations

**Contrat le plus dangereux identifié :**
- Contrat 554 — Prévoyance | Paris | BTP
- Probabilité de déclenchement : 77.1%
- Prime pure ML : 89 806 euros

## Dataset synthétique

Les données réelles de portefeuille réassurance prévoyance sont confidentielles. Ce projet utilise un dataset synthétique généré avec des paramètres actuariellement cohérents, conformément à la pratique standard de la recherche actuarielle.

**Paramètres de génération :**
- 5 000 contrats sur 5 ans (2016-2020)
- GPD calibrée : xi = 0.4, sigma = 80 000 euros (valeurs issues de la littérature Cat)
- Probabilités de sinistre : modèle additif avec drivers zone, branche, BTP, historique
- Taux de déclenchement XS : 11.7%

Le modèle peut être appliqué sur de vraies données en remplaçant uniquement le fichier dataset_xs_predict.csv.

## Tech stack

| Composant | Outil | Rôle |
|-----------|-------|------|
| Simulation | NumPy + SciPy (genpareto) | Monte Carlo + GPD |
| ML Fréquence | XGBoost 2.0 | Classification binaire |
| ML Sévérité | LightGBM 4.0 | Régression Huber |
| Optimisation | Optuna | Hyperparamètres |
| Explicabilité | SHAP | Décomposition des prédictions |
| Dashboard | Streamlit | Interface souscripteur |
| Graphiques | Matplotlib | Visualisations |

## Installation locale

Cloner le repo :

    git clone https://github.com/EddyMroizi/xs-predict.git
    cd xs-predict
    pip install -r requirements.txt

Lancer le pipeline complet (étapes 0 à 4) :

    python reservenet.py

Lancer le dashboard :

    streamlit run streamlit_app.py

## Résultats clés

- **Prime pure MC** : 13.56 M€/an pour un traité 800k XS 200k
- **VaR 99%** : 18.85 M€ — pire cas centennale
- **Structure optimale** : XS Cat — variance S/P 2x inférieure au QP
- **AUC-ROC** : 0.62 — signal prédictif capté sur données synthétiques
- **A/E ratio** : 0.961 — sévérité bien calibrée

## Améliorations futures

- Copule de Clayton pour modéliser la dépendance entre sinistres d'un même événement catastrophique
- Données externes (indices climatiques, NAO) comme features ML
- Export PDF automatique de la note de tarification
- Calibration GPD par Maximum de Vraisemblance pénalisé (MLE régularisé)

## Auteur

**Eddy Mroizi** — Chargé d'études actuarielles

[LinkedIn](https://www.linkedin.com/in/eddy-mroizi-964653265/) · [GitHub](https://github.com/EddyMroizi)