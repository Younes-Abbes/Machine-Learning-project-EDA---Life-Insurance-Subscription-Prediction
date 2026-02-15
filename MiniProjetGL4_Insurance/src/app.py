"""
Streamlit Application for Life Insurance Subscription Prediction
================================================================
A user-friendly web interface for predicting customer subscription probability.

Author: GL4 Data Mining Mini-Project Team
Date: 2026

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Life Insurance Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model and preprocessing artifacts."""
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
        scaler_path = Path(__file__).parent.parent / 'data' / 'processed' / 'scaler.pkl'
        features_path = Path(__file__).parent.parent / 'data' / 'processed' / 'feature_names.pkl'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        
        return model, scaler, feature_names, True
    except FileNotFoundError as e:
        return None, None, None, False


def preprocess_input(input_data: dict, scaler, feature_names: list) -> np.ndarray:
    """
    Preprocess user input for prediction.
    
    Parameters:
    -----------
    input_data : dict
        Raw user input data
    scaler : StandardScaler
        Fitted scaler for numerical features
    feature_names : list
        List of feature names in correct order
        
    Returns:
    --------
    np.ndarray
        Preprocessed input ready for prediction
    """
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode Gender: Female=0, Male=1
    df['Gender'] = 1 if input_data['Gender'] == 'Male' else 0
    
    # Encode Vehicle_Age
    vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    df['Vehicle_Age'] = vehicle_age_map[input_data['Vehicle_Age']]
    
    # Encode Vehicle_Damage
    df['Vehicle_Damage'] = 1 if input_data['Vehicle_Damage'] == 'Yes' else 0
    
    # Ensure correct column order
    df = df[feature_names]
    
    # Scale numerical features
    numeric_cols = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df.values


def create_probability_chart(prob_no: float, prob_yes: float):
    """Create a bar chart showing prediction probabilities."""
    fig = go.Figure(data=[
        go.Bar(
            x=['Non-souscription', 'Souscription'],
            y=[prob_no * 100, prob_yes * 100],
            marker_color=['#dc3545', '#28a745'],
            text=[f'{prob_no*100:.1f}%', f'{prob_yes*100:.1f}%'],
            textposition='outside',
            textfont=dict(size=16, color='black')
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Probabilités de Prédiction',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        yaxis_title='Probabilité (%)',
        yaxis=dict(range=[0, 100]),
        xaxis_title='Résultat',
        showlegend=False,
        height=400,
        plot_bgcolor='white'
    )
    
    return fig


def create_gauge_chart(probability: float):
    """Create a gauge chart for subscription probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Souscription", 'font': {'size': 20}},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#f8d7da'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#d4edda'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=350)
    
    return fig


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Prédiction de Souscription Assurance Vie</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Mini-Projet GL4 Data Mining - INSAT 2026</p>', 
                unsafe_allow_html=True)
    
    # Load model and artifacts
    model, scaler, feature_names, model_loaded = load_model_and_artifacts()
    
    if not model_loaded:
        st.error("""
        ⚠️ **Modèle non chargé!**
        
        Veuillez d'abord exécuter le notebook de modélisation pour entraîner et sauvegarder le modèle.
        
        1. Ouvrir `notebooks/modeling.ipynb`
        2. Exécuter toutes les cellules
        3. Relancer cette application
        """)
        
        st.info("""
        **Mode Démonstration**
        
        En attendant, vous pouvez explorer l'interface avec des prédictions simulées.
        """)
        
        demo_mode = True
    else:
        demo_mode = False
        st.success("✅ Modèle chargé avec succès!")
    
    st.markdown("---")
    
    # Sidebar - Input Form
    st.sidebar.header("📝 Informations Client")
    st.sidebar.markdown("Remplissez les informations ci-dessous:")
    
    # Demographics
    st.sidebar.subheader("👤 Données Démographiques")
    
    gender = st.sidebar.selectbox(
        "Genre",
        options=['Male', 'Female'],
        help="Genre du client"
    )
    
    age = st.sidebar.slider(
        "Âge",
        min_value=18,
        max_value=85,
        value=35,
        help="Âge du client (18-85 ans)"
    )
    
    region_code = st.sidebar.slider(
        "Code Région",
        min_value=0,
        max_value=52,
        value=28,
        help="Code de la région de résidence"
    )
    
    # Vehicle Information
    st.sidebar.subheader("🚗 Informations Véhicule")
    
    driving_license = st.sidebar.selectbox(
        "Permis de Conduire",
        options=[1, 0],
        format_func=lambda x: "Oui" if x == 1 else "Non",
        help="Le client possède-t-il un permis de conduire?"
    )
    
    vehicle_age = st.sidebar.selectbox(
        "Âge du Véhicule",
        options=['< 1 Year', '1-2 Year', '> 2 Years'],
        help="Âge du véhicule du client"
    )
    
    vehicle_damage = st.sidebar.selectbox(
        "Véhicule Endommagé",
        options=['Yes', 'No'],
        format_func=lambda x: "Oui" if x == "Yes" else "Non",
        help="Le véhicule a-t-il déjà été endommagé?"
    )
    
    # Insurance Information
    st.sidebar.subheader("📋 Informations Assurance")
    
    previously_insured = st.sidebar.selectbox(
        "Déjà Assuré",
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
        help="Le client était-il précédemment assuré?"
    )
    
    annual_premium = st.sidebar.number_input(
        "Prime Annuelle (€)",
        min_value=2000,
        max_value=100000,
        value=30000,
        step=1000,
        help="Prime annuelle payée par le client"
    )
    
    policy_sales_channel = st.sidebar.slider(
        "Canal de Vente",
        min_value=1,
        max_value=163,
        value=26,
        help="Code du canal de vente (agent, mail, etc.)"
    )
    
    vintage = st.sidebar.slider(
        "Ancienneté (jours)",
        min_value=10,
        max_value=300,
        value=150,
        help="Nombre de jours depuis l'association avec l'entreprise"
    )
    
    # Prediction Button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Prédire la Souscription", use_container_width=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Résumé des Informations")
        
        # Display input summary
        input_data = {
            'Gender': gender,
            'Age': age,
            'Driving_License': driving_license,
            'Region_Code': region_code,
            'Previously_Insured': previously_insured,
            'Vehicle_Age': vehicle_age,
            'Vehicle_Damage': vehicle_damage,
            'Annual_Premium': annual_premium,
            'Policy_Sales_Channel': policy_sales_channel,
            'Vintage': vintage
        }
        
        # Create summary cards
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("""
            <div class="metric-card">
                <h4>👤 Profil Client</h4>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Genre:** {'Homme' if gender == 'Male' else 'Femme'}")
            st.write(f"**Âge:** {age} ans")
            st.write(f"**Région:** {region_code}")
            st.write(f"**Ancienneté:** {vintage} jours")
        
        with summary_col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🚗 Véhicule & Assurance</h4>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Permis:** {'Oui' if driving_license == 1 else 'Non'}")
            st.write(f"**Âge véhicule:** {vehicle_age}")
            st.write(f"**Endommagé:** {'Oui' if vehicle_damage == 'Yes' else 'Non'}")
            st.write(f"**Prime:** {annual_premium:,} €")
    
    with col2:
        st.subheader("🎯 Résultat de Prédiction")
        
        if predict_button:
            with st.spinner("Analyse en cours..."):
                if demo_mode:
                    # Demo mode: simulate prediction
                    # Simple heuristic for demo
                    base_prob = 0.12
                    if vehicle_damage == 'Yes':
                        base_prob += 0.25
                    if previously_insured == 0:
                        base_prob += 0.15
                    if 25 <= age <= 50:
                        base_prob += 0.10
                    if vehicle_age == '1-2 Year':
                        base_prob += 0.05
                    
                    prob_yes = min(base_prob + np.random.uniform(-0.05, 0.05), 0.95)
                    prob_no = 1 - prob_yes
                else:
                    # Real prediction
                    X_input = preprocess_input(input_data, scaler, feature_names)
                    probabilities = model.predict_proba(X_input)[0]
                    prob_no, prob_yes = probabilities[0], probabilities[1]
                
                # Display gauge chart
                gauge_fig = create_gauge_chart(prob_yes)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Recommendation
                if prob_yes >= 0.6:
                    st.markdown("""
                    <div class="prediction-box success-box">
                        <h2>✅ Souscription Recommandée!</h2>
                        <p>Ce client a une forte probabilité de souscrire à l'assurance vie.</p>
                        <p><strong>Action suggérée:</strong> Contacter le client avec une offre personnalisée.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prob_yes >= 0.4:
                    st.markdown("""
                    <div class="prediction-box warning-box">
                        <h2>⚠️ Souscription Possible</h2>
                        <p>Ce client pourrait être intéressé par l'assurance vie.</p>
                        <p><strong>Action suggérée:</strong> Proposer une offre d'essai ou des informations supplémentaires.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-box danger-box">
                        <h2>❌ Souscription Peu Probable</h2>
                        <p>Ce client a une faible probabilité de souscrire.</p>
                        <p><strong>Action suggérée:</strong> Concentrer les efforts sur d'autres prospects.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability bar chart
                st.markdown("---")
                bar_fig = create_probability_chart(prob_no, prob_yes)
                st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("👈 Remplissez les informations client dans la barre latérale et cliquez sur 'Prédire'")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Mini-Projet GL4 - Data Mining</strong></p>
        <p>INSAT - Institut National des Sciences Appliquées et de Technologie</p>
        <p>2026 - Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
