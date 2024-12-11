import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

def mark_invalid_ages_as_nan(X):
    X[X <= 0] = pd.NA
    return X

def encode_classes_binary(y, class_mapping):
    y_encoded = y.map(class_mapping)
    y_encoded_binary = y_encoded.apply(lambda x: f"{x:02b}")
    return y_encoded_binary

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            if X[col].dtype == 'object' or pd.api.types.is_bool_dtype(X[col]):
                le.fit(X[col])
                self.label_encoders[col] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X_copy[col] = le.transform(X[col])
        return X_copy

def preprocess_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('age_marker', FunctionTransformer(mark_invalid_ages_as_nan, validate=False)),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('to_integer', FunctionTransformer(lambda X: X.astype(int))),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('to_dataframe', FunctionTransformer(lambda X: pd.DataFrame(X))),
        ('label_encoder', CustomLabelEncoder()),
        ('scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    X_preprocessed = pipeline.fit_transform(X)
    class_mapping = {'RE': 0, 'RF': 1, 'RM': 2}
    y_encoded_binary = encode_classes_binary(y, class_mapping)
    le_y = LabelEncoder().fit(y)
    y_preprocessed = y_encoded_binary
    return X_preprocessed, y_preprocessed, le_y, pipeline, preprocessor

def calculate_risk_coefficient(model, X_preprocessed_df, y_preprocessed_df, data_df, feature_name, feature_value):
    # Convertir feature_value au même type que dans data_df
    feature_value = pd.Series([feature_value]).astype(data_df[feature_name].dtype)[0]

    # Trouver l'indice de feature_value dans les données originales
    if feature_value in data_df[feature_name].values:
        index = data_df[data_df[feature_name] == feature_value].index[0]
    else:
        raise ValueError(f"La valeur '{feature_value}' n'a pas été trouvée dans la colonne '{feature_name}' de data_df")

    # Prédire les probabilités en utilisant le modèle de stacking
    probas = model.predict_proba([X_preprocessed_df.iloc[index]])

    # Trouver l'indice de la classe avec la probabilité la plus élevée
    predicted_class_index = np.argmax(probas[0])

    # Trouver la classe prédite
    predicted_class = model.classes_[predicted_class_index]

    # Calculer le coefficient de risque en pourcentage basé sur la proportion de la classe prédite
    risk_percentage = probas[0][predicted_class_index] * 100

    # Déterminer le niveau de risque associé en fonction du coefficient de risque
    if predicted_class == '00':
        risk_suggestion = 'RE (Risque Élevé)'
    elif predicted_class == '01':
        risk_suggestion = 'RF (Risque Faible)'
    elif predicted_class == '10':
        risk_suggestion = 'RM (Risque Moyen)'
    else:
        risk_suggestion = 'Inconnu'

    return risk_percentage, risk_suggestion

def predict_risk_score(model, preprocess_pipeline, feature_dict):
    # Convertir le dictionnaire en DataFrame
    input_df = pd.DataFrame([feature_dict])

    # Prétraiter les données
    preprocessed_input = preprocess_pipeline.transform(input_df)

    # Prédire les probabilités pour chaque classe
    risk_probas = model.predict_proba(preprocessed_input)[0]

    # Mapper les probabilités aux labels
    risk_mapping = {'00': 'RE', '01': 'RF', '10': 'RM'}
    
    # Convertir les labels des classes du modèle en chaînes binaires pour correspondre au mappage
    class_labels = model.classes_.astype(str)  # Convertir les classes en chaînes

    # Créer un dictionnaire avec les probabilités pour chaque classe
    risk_percentages = {risk_mapping[class_labels[i]]: prob * 100 for i, prob in enumerate(risk_probas)}

    # Déterminer la classe prédite
    predicted_class_index = risk_probas.argmax()
    predicted_risk = risk_mapping[class_labels[predicted_class_index]]

    # Définir des poids pour chaque classe (ajustez selon le projet)
    weights = {'RE': 0.1, 'RF': 0.1, 'RM': 0.5}

    # Calculer le score de risque global
    total_weighted_score = sum(weights[risk_mapping[class_labels[i]]] * risk_probas[i] for i in range(len(risk_probas)))
    total_prob = sum(risk_probas)
    risk_coefficient = (total_weighted_score / total_prob) * 100  # Normaliser pour être dans la plage 0-100%

    return predicted_risk, risk_percentages, risk_coefficient
