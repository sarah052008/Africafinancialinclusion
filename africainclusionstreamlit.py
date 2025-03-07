import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Titre de l'application
st.title("Financial Inclusion Prediction App")
st.write("Cette application prédit si un individu est susceptible d'avoir ou d'utiliser un compte bancaire.")

# Charger les modèles
@st.cache_resource  # Cache les modèles pour éviter de les recharger à chaque interaction
def load_models():
    logistic_model = joblib.load('logistic_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    mlp_model = joblib.load('mlp_model.pkl')
    return logistic_model, rf_model, mlp_model

logistic_model, rf_model, mlp_model = load_models()

# Interface utilisateur pour choisir le modèle
st.sidebar.header("Choix du Modèle")
model_choice = st.sidebar.selectbox(
    "Sélectionnez un modèle",
    ["Logistic Regression", "Random Forest", "MLP Classifier"]
)

# Charger le modèle sélectionné
if model_choice == "Logistic Regression":
    model = logistic_model
elif model_choice == "Random Forest":
    model = rf_model
elif model_choice == "MLP Classifier":
    model = mlp_model

# Interface utilisateur pour saisir les caractéristiques
st.sidebar.header("Saisissez les caractéristiques de l'individu")

def user_input_features():
    # Caractéristiques catégorielles
    country = st.sidebar.selectbox("Pays", ["Rwanda", "Tanzania", "Kenya", "Uganda"])
    location_type = st.sidebar.selectbox("Type de localisation", ["Rural", "Urban"])
    cellphone_access = st.sidebar.selectbox("Accès à un téléphone portable", ["Yes", "No"])
    relationship_with_head = st.sidebar.selectbox("Relation avec le chef de famille", [
        "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
    ])
    marital_status = st.sidebar.selectbox("Statut matrimonial", [
        "Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Dont know"
    ])
    education_level = st.sidebar.selectbox("Niveau d'éducation", [
        "Primary education", "No formal education", "Secondary education", "Tertiary education", 
        "Vocational/Specialised training", "Other/Dont know/RTA"
    ])
    job_type = st.sidebar.selectbox("Type d'emploi", [
        "Self employed", "Informally employed", "Farming and Fishing", "Remittance Dependent", 
        "Other Income", "Formally employed Private", "Government Dependent", "No Income"
    ])
    gender_of_respondent = st.sidebar.selectbox("Genre", ["Female", "Male"])

    # Caractéristiques numériques
    year = st.sidebar.number_input("Année", min_value=1900, max_value=2100, value=2023)
    age_of_respondent = st.sidebar.number_input("Âge de l'individu", min_value=0, max_value=120, value=30)
    household_size = st.sidebar.number_input("Taille du ménage", min_value=1, max_value=20, value=4)

    # Encodage des valeurs catégorielles
    country_encoded = {"Rwanda": 0, "Tanzania": 1, "Kenya": 2, "Uganda": 3}[country]
    location_type_encoded = {"Rural": 0, "Urban": 1}[location_type]
    cellphone_access_encoded = {"Yes": 1, "No": 0}[cellphone_access]
    relationship_with_head_encoded = {
        "Head of Household": 0, "Spouse": 1, "Child": 2, "Parent": 3, "Other relative": 4, "Other non-relatives": 5
    }[relationship_with_head]
    marital_status_encoded = {
        "Married/Living together": 0, "Single/Never Married": 1, "Widowed": 2, "Divorced/Seperated": 3, "Dont know": 4
    }[marital_status]
    education_level_encoded = {
        "Primary education": 0, "No formal education": 1, "Secondary education": 2, 
        "Tertiary education": 3, "Vocational/Specialised training": 4, "Other/Dont know/RTA": 5
    }[education_level]
    job_type_encoded = {
        "Self employed": 0, "Informally employed": 1, "Farming and Fishing": 2, 
        "Remittance Dependent": 3, "Other Income": 4, "Formally employed Private": 5, 
        "Government Dependent": 6, "No Income": 7
    }[job_type]
    gender_of_respondent_encoded = {"Female": 0, "Male": 1}[gender_of_respondent]

    # Créer un dictionnaire avec les données encodées et numériques
    data = {
        "country": country_encoded,
        "location_type": location_type_encoded,
        "cellphone_access": cellphone_access_encoded,
        "relationship_with_head": relationship_with_head_encoded,
        "marital_status": marital_status_encoded,
        "education_level": education_level_encoded,
        "job_type": job_type_encoded,
        "gender_of_respondent": gender_of_respondent_encoded,
        "year": year,
        "age_of_respondent": age_of_respondent,
        "household_size": household_size,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Saisie des caractéristiques
input_df = user_input_features()

# Afficher les caractéristiques saisies
st.subheader("Caractéristiques Saisies")
st.write(input_df)

# Faire la prédiction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Afficher les résultats de la prédiction
st.subheader("Résultat de la Prédiction")
st.write("L'individu est susceptible d'avoir un compte bancaire :", "Oui" if prediction[0] == 1 else "Non")

st.subheader("Probabilités de Prédiction")
st.write(f"- Probabilité de ne pas avoir de compte bancaire : {prediction_proba[0][0]:.2f}")
st.write(f"- Probabilité d'avoir un compte bancaire : {prediction_proba[0][1]:.2f}")

# Visualisations
st.header("Visualisations")

# Matrice de confusion (exemple avec des données fictives)
st.subheader("Matrice de Confusion")
# Générer des prédictions sur un ensemble de test fictif
y_true = np.random.randint(0, 2, 100)  # Données réelles fictives
y_pred = np.random.randint(0, 2, 100)  # Prédictions fictives
conf_matrix = confusion_matrix(y_true, y_pred)

# Afficher la matrice de confusion
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
st.pyplot(fig)

# Rapport de classification
st.subheader("Rapport de Classification")
st.text(classification_report(y_true, y_pred))

# Histogramme des prédictions
st.subheader("Distribution des Prédictions")
fig, ax = plt.subplots()
sns.histplot(y_pred, kde=True, ax=ax)
ax.set_xlabel("Prédictions")
ax.set_ylabel("Fréquence")
st.pyplot(fig)