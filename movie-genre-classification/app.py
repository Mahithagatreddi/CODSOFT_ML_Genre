import streamlit as st
import joblib
import re
import string
import os
import time
import numpy as np
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Cinematic Genre Predictor",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Advanced CSS (Glassmorphism & Background) ---
st.markdown("""
<style>
    /* Background Image - Mysterious dark fog/smoke */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?q=80&w=1974&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Dark Overlay with a slight purple tint for mystery */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(15, 5, 25, 0.6); /* Lowered opacity to show background */
        z-index: 0;
    }
    
    /* Glassmorphism applied to the main centered container */
    .block-container {
        position: relative;
        z-index: 1;
        background: rgba(20, 10, 30, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 0, 85, 0.2);
        border-radius: 20px;
        padding: 4rem !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
        margin-top: 4vh !important;
        margin-bottom: 4vh !important;
    }
    
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Title */
    .cinematic-title {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff0055, #6a0dad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-shadow: 0px 4px 20px rgba(255, 0, 85, 0.3);
        font-family: 'Georgia', serif;
        letter-spacing: 2px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #c8b6d9;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
        letter-spacing: 3px;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(20, 10, 30, 0.6) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 0, 85, 0.3) !important;
        border-radius: 10px;
        font-size: 1.1rem;
        padding: 15px;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff0055 !important;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.4) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #6a0dad 0%, #ff0055 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 15px 30px;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(106, 13, 173, 0.5);
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(255, 0, 85, 0.6);
        color: white;
    }
    
    /* Result Header */
    .result-header {
        text-align: center;
        font-size: 1.5rem;
        color: white;
        margin-top: 30px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Logic Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_models():
    model_path = 'models/logistic_regression.joblib'
    vectorizer_path = 'models/vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

# --- Main Interface ---

st.markdown('<p class="cinematic-title">THE ENIGMA ENGINE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">UNCOVERING THE HIDDEN GENRES</p>', unsafe_allow_html=True)

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("⚠️ Model files not found. Please train the model first.")
else:
        plot_summary = st.text_area(
            label="Plot Summary", 
            label_visibility="collapsed",
            height=200, 
            placeholder="Paste a movie plot summary here to reveal its true genre..."
        )
        
        # Add some vertical spacing
        st.write("")
        
        # Center the button using 3 equal columns
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col2:
            predict_button = st.button("INITIATE ANALYSIS")
            
        if predict_button:
            if not plot_summary.strip():
                st.warning("Please enter a plot summary before analyzing.")
            else:
                progress_bar = st.progress(0)
                time.sleep(0.2)
                progress_bar.progress(30)
                
                cleaned_text = clean_text(plot_summary)
                features = vectorizer.transform([cleaned_text])
                time.sleep(0.3)
                progress_bar.progress(70)
                
                # We use predict_proba instead of just predict!
                try:
                    probabilities = model.predict_proba(features)[0]
                    classes = model.classes_
                    
                    # Sort by probability
                    sorted_indices = np.argsort(probabilities)[::-1]
                    top_classes = classes[sorted_indices][:3]
                    top_probs = probabilities[sorted_indices][:3]
                    
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_bar.empty()
                    
                    # Highest Prediction Text
                    top_genre = top_classes[0].upper()
                    st.markdown(f'<p class="result-header">The Truth Is Revealed: <span style="color:#ff0055;">{top_genre}</span></p>', unsafe_allow_html=True)
                    
                    # Plotly Interactive Chart
                    df_probs = pd.DataFrame({
                        'Genre': top_classes,
                        'Probability': top_probs * 100
                    })
                    
                    fig = px.bar(
                        df_probs, 
                        x='Probability', 
                        y='Genre', 
                        orientation='h',
                        color='Probability',
                        color_continuous_scale=['#6a0dad', '#ff0055']
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=14),
                        xaxis=dict(title='Confidence (%)', range=[0, 100], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(title='', autorange="reversed"),
                        margin=dict(l=0, r=0, t=30, b=0),
                        coloraxis_showscale=False,
                        height=250
                    )
                    
                    # Add data labels
                    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.balloons()
                    
                except AttributeError:
                    # Fallback if model doesn't support predict_proba
                    prediction = model.predict(features)[0]
                    progress_bar.empty()
                    st.markdown(f'<p class="result-header">Predicted Genre: <span style="color:#00d2ff;">{prediction.upper()}</span></p>', unsafe_allow_html=True)
