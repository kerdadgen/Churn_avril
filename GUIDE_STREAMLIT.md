# 🚀 Guide d'utilisation de l'interface Streamlit

## Installation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Générer le modèle (si ce n'est pas fait)
```bash
python train.py
```

Cela créera le fichier `churn_regression_log.pkl` nécessaire.

## Lancer l'interface

### Option 1 : Interface Streamlit (RECOMMANDÉE) 🎨
```bash
streamlit run streamlit_app.py
```

L'interface s'ouvrira automatiquement dans votre navigateur sur `http://localhost:8501`

**Fonctionnalités:**
- 📋 Formulaire intuitif pour entrer les données client
- 🎯 Prédiction instantanée du risque de churn
- 📊 Visualisation graphique des probabilités
- ✅ Design moderne et responsive

### Option 2 : API Flask (pour intégration API)
```bash
python app.py
```

L'API sera disponible sur `http://localhost:5000`

**Endpoints:**
- `GET /health` - Vérifier l'état du service
- `POST /predict` - Faire une prédiction

Exemple avec curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "Total_Purchase": 5000, "Years": 3, "Num_Sites": 5}'
```

## Utilisation de l'interface Streamlit

### Étape 1 : Accéder à l'interface
Une fois lancé, l'interface s'ouvre automatiquement. Vous verrez:
- **Colonne gauche:** Formulaire d'entrée
- **Colonne droite:** Résultats de la prédiction
- **Barre latérale:** Informations et guide

### Étape 2 : Remplir le formulaire
Entrez les informations du client:
- **Âge:** Entre 18 et 100 ans
- **Total des achats:** Montant en euros
- **Années de relation:** Nombre d'années depuis l'enregistrement
- **Nombre de sites:** Nombre de sites du client

### Étape 3 : Faire la prédiction
Cliquez sur le bouton **"🔮 Faire la prédiction"**

### Étape 4 : Interpréter les résultats

#### Résultats possibles:

**🟢 CLIENT FIDÈLE (Probabilité de départ < 50%)**
- Le client a peu de risque de partir
- Probabilité de rester élevée

**🔴 RISQUE DE CHURN DÉTECTÉ (Probabilité de départ ≥ 50%)**
- Le client a un risque significatif de départ
- Recommandation: engager une action de rétention

## Caractéristiques du modèle

- **Algorithme:** Régression Logistique
- **Nombre de features:** 4
  - Age
  - Total_Purchase
  - Years
  - Num_Sites
- **Type de prédiction:** Classification binaire (Churn/Non-Churn)

## Architecture

```
Churn_avril/
├── app.py                 # API Flask (optionnelle)
├── streamlit_app.py       # Interface Streamlit (PRINCIPALE)
├── train.py              # Script d'entraînement du modèle
├── requirements.txt      # Dépendances Python
├── README.md             # Documentation principale
├── GUIDE_STREAMLIT.md    # Ce fichier
├── churn_regression_log.pkl  # Modèle entraîné
└── data/
    └── customer_churn.csv    # Dataset d'entraînement
```

## Dépannage

### Erreur: "Modèle non trouvé"
**Solution:** Exécutez `python train.py` pour générer le modèle

### Erreur: "Streamlit not found"
**Solution:** Installez Streamlit avec `pip install streamlit`

### L'interface ne s'ouvre pas
**Solution:** Accédez manuellement à `http://localhost:8501` dans votre navigateur

## Tips & Tricks

✨ **Tester différents scénarios:** L'interface permet de tester rapidement plusieurs combinaisons de paramètres

🔄 **Accès rapide:** Utilisez l'historique du navigateur pour revenir aux prédictions précédentes

💾 **Export:** Vous pouvez prendre des captures d'écran des résultats

## Performance

- ⚡ Prédictions instantanées (<100ms)
- 📊 Interface responsive
- 🔁 Rechargement rapide après chaque prédiction

---

**Bon usage! 🎉**
