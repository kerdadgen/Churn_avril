import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de Churn Client",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter du CSS personnalisé
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 2rem 0;
        }
        .churn-yes {
            background-color: #ffcccc;
            color: #cc0000;
            border: 2px solid #cc0000;
        }
        .churn-no {
            background-color: #ccffcc;
            color: #00cc00;
            border: 2px solid #00cc00;
        }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle
MODEL_PATH = "churn_regression_log.pkl"

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"❌ Le fichier modèle '{MODEL_PATH}' n'a pas été trouvé.")
        st.info("Veuillez exécuter `python train.py` d'abord pour générer le modèle.")
        return None

# Titre principal
st.title("🎯 Prédicteur de Churn Client")
st.markdown("---")

# Charger le modèle
model = load_model()

if model is not None:
    # Créer deux colonnes
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📋 Entrez les informations du client")
        
        # Formulaire d'entrée
        with st.form("prediction_form"):
            age = st.number_input(
                "Âge du client (années)",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
                help="L'âge du client en années"
            )
            
            total_purchase = st.number_input(
                "Total des achats (€)",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                help="Montant total des achats du client"
            )
            
            years = st.number_input(
                "Années de relation",
                min_value=0,
                max_value=50,
                value=3,
                step=1,
                help="Nombre d'années depuis l'enregistrement du client"
            )
            
            num_sites = st.number_input(
                "Nombre de sites",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                help="Nombre de sites où le client exerce ses activités"
            )
            
            submitted = st.form_submit_button(
                "🔮 Faire la prédiction",
                use_container_width=True
            )
    
    with col2:
        st.subheader("📊 Résultat de la prédiction")
        
        if submitted:
            # Préparer les données
            features = pd.DataFrame({
                "Age": [age],
                "Total_Purchase": [total_purchase],
                "Years": [years],
                "Num_Sites": [num_sites]
            })
            
            # Faire la prédiction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Afficher le résultat
            churn_prob = probability[1] * 100
            no_churn_prob = probability[0] * 100
            
            # Boîte de résultat
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="prediction-box churn-yes">
                    ⚠️ RISQUE DE CHURN DÉTECTÉ
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.error(f"Probabilité de départ : **{churn_prob:.1f}%**")
            else:
                st.markdown(
                    f"""
                    <div class="prediction-box churn-no">
                    ✅ CLIENT FIDÈLE
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.success(f"Probabilité de départ : **{churn_prob:.1f}%**")
            
            st.markdown("---")
            
            # Détails des probabilités
            col_prob1, col_prob2 = st.columns(2)
            
            with col_prob1:
                st.metric(
                    "Probabilité de départ",
                    f"{churn_prob:.1f}%",
                    delta=None
                )
            
            with col_prob2:
                st.metric(
                    "Probabilité de rester",
                    f"{no_churn_prob:.1f}%",
                    delta=None
                )
            
            # Graphique visuel
            st.markdown("### Analyse visuelle")
            
            # Créer un graphique Plotly avec couleurs personnalisées
            fig = go.Figure(data=[
                go.Bar(
                    x=['Départ', 'Fidélité'],
                    y=[churn_prob, no_churn_prob],
                    marker=dict(
                        color=['#ff4444', '#44ff44'],
                        line=dict(color='#333333', width=2)
                    ),
                    text=[f'{churn_prob:.1f}%', f'{no_churn_prob:.1f}%'],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Probabilité: %{y:.1f}%<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Distribution des probabilités",
                xaxis_title="Scénario",
                yaxis_title="Probabilité (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Résumé des données
            st.markdown("### 📝 Résumé du client")
            summary_df = pd.DataFrame({
                'Paramètre': ['Âge', 'Total des achats', 'Années de relation', 'Nombre de sites'],
                'Valeur': [f"{age} ans", f"€{total_purchase:,.2f}", f"{years} ans", f"{num_sites} sites"]
            })
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Barre latérale avec informations
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ À propos")
    st.sidebar.info(
        """
        **Prédicteur de Churn Client**
        
        Cet outil utilise un modèle de **Régression Logistique** 
        entraîné sur les données de churn clients.
        
        **Objectif:** Prédire la probabilité qu'un client 
        quitte le service en fonction de ses caractéristiques.
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Guide d'utilisation")
    st.sidebar.write(
        """
        1. **Entrez les informations du client** dans le formulaire
        2. **Cliquez sur "Faire la prédiction"**
        3. **Consultez les résultats** avec les probabilités
        
        **Interprétation:**
        - **Risque élevé (>50%)** → Action recommandée
        - **Risque faible (<50%)** → Client généralement fidèle
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Modèle")
    st.sidebar.write(
        """
        - **Type:** Régression Logistique
        - **Features:** 4
        - **Modèle:** churn_regression_log.pkl
        """
    )

else:
    st.error("Le modèle n'a pas pu être chargé. Veuillez vérifier les fichiers.")
