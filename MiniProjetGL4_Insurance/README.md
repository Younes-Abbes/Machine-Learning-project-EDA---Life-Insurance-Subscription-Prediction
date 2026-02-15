# 🏥 Mini-Projet GL4 - Prédiction de Souscription à l'Assurance Vie

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 📋 Description

Projet de Data Mining pour la prédiction de souscription à l'assurance vie utilisant des techniques de Machine Learning. Ce projet suit la méthodologie CRISP-DM et comprend une analyse exploratoire complète, la comparaison de 5 modèles de classification, et le déploiement d'une application web interactive.

**Institut:** INSAT - Institut National des Sciences Appliquées et de Technologie  
**Cours:** GL4 - Data Mining  
**Année:** 2026

---

## 🎯 Objectifs

- Prédire si un client souscrira à une assurance vie (classification binaire)
- Analyser les facteurs influençant la décision de souscription
- Comparer différents algorithmes de Machine Learning
- Déployer une application de prédiction conviviale

---

## 📊 Dataset

Le dataset provient de [Kaggle - Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

| Statistique | Valeur |
|-------------|--------|
| Nombre d'observations | 381,109 |
| Nombre de features | 11 |
| Variable cible | Response (0/1) |
| Déséquilibre de classes | 87.7% / 12.3% |

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| Gender | Catégoriel | Genre du client |
| Age | Numérique | Âge du client |
| Driving_License | Binaire | Possession d'un permis |
| Region_Code | Numérique | Code région |
| Previously_Insured | Binaire | Déjà assuré |
| Vehicle_Age | Catégoriel | Âge du véhicule |
| Vehicle_Damage | Catégoriel | Véhicule endommagé |
| Annual_Premium | Numérique | Prime annuelle |
| Policy_Sales_Channel | Numérique | Canal de vente |
| Vintage | Numérique | Ancienneté (jours) |
| **Response** | **Binaire** | **Cible: Souscription** |

---

## 🏗️ Structure du Projet

```
MiniProjetGL4_Insurance/
│
├── 📁 data/
│   ├── train.csv                 # Dataset original
│   └── processed/                # Données prétraitées
│       ├── X_train.pkl, X_test.pkl
│       ├── y_train.pkl, y_test.pkl
│       ├── scaler.pkl
│       └── feature_names.pkl
│
├── 📁 notebooks/
│   ├── eda.ipynb                 # Analyse exploratoire
│   └── modeling.ipynb            # Modélisation & évaluation
│
├── 📁 src/
│   ├── preprocess.py             # Module de prétraitement
│   ├── models.py                 # Module de modélisation
│   └── app.py                    # Application Streamlit
│
├── 📁 models/
│   └── best_model.pkl            # Meilleur modèle sauvegardé
│
├── 📁 figs/
│   ├── eda_*.png                 # Graphiques EDA
│   └── model_*.png               # Graphiques modélisation
│
├── 📁 report/
│   ├── report.tex                # Rapport LaTeX
│   └── report.pdf                # Rapport PDF (généré)
│
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- pip (gestionnaire de packages Python)

### Étapes d'installation

1. **Cloner/Extraire le projet**
   ```bash
   cd MiniProjetGL4_Insurance
   ```

2. **Créer un environnement virtuel (recommandé)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Utilisation

### 1. Analyse Exploratoire (EDA)

Ouvrir et exécuter le notebook Jupyter:
```bash
jupyter notebook notebooks/eda.ipynb
```

Ce notebook contient:
- Chargement et exploration des données
- Analyse de la qualité des données
- Visualisations univariées et bivariées
- Matrice de corrélation
- Insights clés

### 2. Modélisation

Exécuter le notebook de modélisation:
```bash
jupyter notebook notebooks/modeling.ipynb
```

Ce notebook inclut:
- Prétraitement complet (outliers, encodage, scaling, SMOTE)
- Entraînement de 5 modèles (Logistic Regression, Random Forest, KNN, XGBoost, SVM)
- Cross-validation 5-fold stratifiée
- Hyperparameter tuning (GridSearchCV)
- Évaluation et comparaison des modèles
- Sauvegarde du meilleur modèle

### 3. Application Streamlit

Lancer l'application de prédiction:
```bash
cd src
streamlit run app.py
```

Ou depuis la racine:
```bash
streamlit run src/app.py
```

L'application sera accessible sur: **http://localhost:8501**

### 4. Générer le Rapport PDF

```bash
cd report
pdflatex report.tex
pdflatex report.tex  # Exécuter 2 fois pour la table des matières
```

---

## 🤖 Modèles Implémentés

| Modèle | Paramètres |
|--------|------------|
| Logistic Regression | C=1.0, solver='liblinear' |
| Random Forest | n_estimators=100, max_depth=10 |
| K-Nearest Neighbors | n_neighbors=5, metric='euclidean' |
| XGBoost | n_estimators=100, learning_rate=0.1, max_depth=5 |
| Support Vector Machine | kernel='rbf', C=1.0 |

---

## 📈 Résultats

### Cross-Validation (5-Fold)

| Modèle | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|---------|
| **XGBoost** | **0.854** | **0.855** | **0.932** |
| Random Forest | 0.850 | 0.851 | 0.929 |
| Logistic Regression | 0.782 | 0.784 | 0.863 |
| SVM | 0.771 | 0.773 | 0.852 |
| KNN | 0.746 | 0.749 | 0.825 |

**Meilleur modèle:** XGBoost (après hyperparameter tuning)

### Variables les Plus Importantes

1. 🥇 Previously_Insured
2. 🥈 Vehicle_Damage
3. 🥉 Policy_Sales_Channel
4. Age
5. Vehicle_Age

---

## 🎨 Captures d'Écran

### EDA - Distribution de la Variable Cible
![Target Distribution](figs/eda_01_target_distribution.png)

### Comparaison des Modèles
![Model Comparison](figs/model_01_cv_comparison.png)

### Courbes ROC
![ROC Curves](figs/model_03_roc_curves.png)

---

## 📚 Dépendances

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
streamlit>=1.28.0
joblib>=1.3.0
openpyxl>=3.1.0
```

---

## 👥 Équipe

- **Projet:** Mini-Projet GL4 Data Mining
- **Institution:** INSAT - Tunis, Tunisie
- **Année:** 2026

---

## 📝 Licence

Ce projet est réalisé dans un cadre académique pour le cours de Data Mining GL4 à l'INSAT.

---

## 🔗 Références

- [Dataset Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)

---

<div align="center">

**⭐ Mini-Projet GL4 - Data Mining - INSAT 2026 ⭐**

</div>
