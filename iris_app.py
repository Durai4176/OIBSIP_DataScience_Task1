import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

st.set_page_config(page_title="Iris Flower Classification", layout="wide")

np.random.seed(42)

@st.cache_data
def load_data():
    csv_path = "C:\\Users\\kumar\\OneDrive\\Desktop\\streamlit\\internship\\oasis\\Iris.csv"
    iris_df = pd.read_csv(csv_path)
    
    if 'Id' in iris_df.columns:
        iris_df = iris_df.drop('Id', axis=1)
    
    feature_names = iris_df.columns[:-1].tolist()
    target_column = 'Species'
    
    X = iris_df[feature_names].values
    y = iris_df[target_column].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_
    
    return X, y_encoded, feature_names, target_names, iris_df, target_column

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    model_accuracies = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy * 100
        trained_models[name] = model
    
    return trained_models, model_accuracies, scaler, X_test, y_test, X_train, y_train

def predict_species(model, scaler, features, target_names):
    features_scaled = scaler.transform([features])
    
    prediction = model.predict(features_scaled)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        proba_df = pd.DataFrame({
            'Species': target_names,
            'Probability': probabilities
        })
        return prediction[0], proba_df
    else:
        return prediction[0], None

def main():
    st.title("🌸 Iris Flower Classification App (CSV)")
    st.markdown("""
    This app classifies Iris flowers into three species (setosa, versicolor, and virginica) based on their measurements.
    You can explore the dataset, visualize the data, and make predictions using different machine learning models.
    """)
    
    X, y, feature_names, target_names, iris_df, target_column = load_data()
    
    trained_models, model_accuracies, scaler, X_test, y_test, X_train, y_train = train_models(X, y)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Model Performance", "🔮 Make Prediction"])
    
    with tab1:
        st.header("Data Exploration")
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of samples: {X.shape[0]}")
            st.write(f"Number of features: {X.shape[1]}")
        with col2:
            st.write(f"Features: {', '.join(feature_names)}")
            st.write(f"Target classes: {', '.join(target_names)}")
        
        st.subheader("First 5 rows of the dataset")
        st.dataframe(iris_df.head())
        
        st.subheader("Basic statistics of the dataset")
        st.dataframe(iris_df.describe())
        
        st.subheader("Data Visualization")
        
        st.write("Pairplot of Iris Dataset Features by Species")
        pairplot_fig = sns.pairplot(iris_df, hue=target_column, markers=['o', 's', 'D'])
        pairplot_fig.fig.suptitle('Pairplot of Iris Dataset Features by Species', y=1.02)
        st.pyplot(pairplot_fig.fig)
        
        st.write("Correlation Heatmap of Iris Features")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(iris_df[feature_names].corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap of Iris Features')
        st.pyplot(fig)
        
        st.write("Feature Distribution by Species")
        feature_option = st.selectbox("Select a feature to visualize:", feature_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=target_column, y=feature_option, data=iris_df, ax=ax)
        plt.title(f'Distribution of {feature_option} by Species')
        st.pyplot(fig)
    
    with tab2:
        st.header("Model Performance")
        
        model_option = st.selectbox("Select a model to evaluate:", list(trained_models.keys()))
        selected_model = trained_models[model_option]
        
        st.subheader(f"{model_option} Performance")
        st.write(f"Accuracy: {model_accuracies[model_option]:.2f}%")
        
        y_pred = selected_model.predict(scaler.transform(X_test))
        
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.title(f'Confusion Matrix - {model_option}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
        st.subheader("Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), ax=ax)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        if model_option == 'Random Forest':
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': selected_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            plt.title('Feature Importance (Random Forest)')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(feature_importance)
    
    with tab3:
        st.header("Make Your Own Prediction")
        st.write("Adjust the sliders to input measurements and get a prediction.")
        
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.slider("Sepal Length (cm)", float(iris_df[feature_names[0]].min()), float(iris_df[feature_names[0]].max()), float(iris_df[feature_names[0]].mean()))
            sepal_width = st.slider("Sepal Width (cm)", float(iris_df[feature_names[1]].min()), float(iris_df[feature_names[1]].max()), float(iris_df[feature_names[1]].mean()))
        with col2:
            petal_length = st.slider("Petal Length (cm)", float(iris_df[feature_names[2]].min()), float(iris_df[feature_names[2]].max()), float(iris_df[feature_names[2]].mean()))
            petal_width = st.slider("Petal Width (cm)", float(iris_df[feature_names[3]].min()), float(iris_df[feature_names[3]].max()), float(iris_df[feature_names[3]].mean()))
        
        input_features = [sepal_length, sepal_width, petal_length, petal_width]
        
        model_option = st.selectbox("Select a model for prediction:", list(trained_models.keys()), key='prediction_model')
        selected_model = trained_models[model_option]
        
        if st.button("Predict"):
            prediction_idx, probabilities = predict_species(selected_model, scaler, input_features, target_names)
            predicted_species = target_names[prediction_idx]
            
            st.success(f"Predicted Species: **{predicted_species}**")
            
            if probabilities is not None:
                st.write("Prediction Probabilities:")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x='Species', y='Probability', data=probabilities, ax=ax)
                plt.title('Prediction Probabilities')
                plt.ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.dataframe(probabilities)
            
            species_name = predicted_species.split('-')[-1].lower()
            st.image(f"https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/main/app_8_classification_iris/images/{species_name}.jpg", 
                     caption=f"Iris {predicted_species}", width=300)

if __name__ == "__main__":
    main()