from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow.models import Variable
import pandas as pd
import numpy as np
from pipeline_functions import preprocess_data, calculate_risk_coefficient, predict_risk_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
import joblib
import os

# Chemin pour sauvegarder le pipeline
PIPELINE_SAVE_PATH = '/home/dhouha/airflow/pipeline/pipeline.joblib'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'kyc_pipeline',
    default_args=default_args,
    description='DAG pour évaluer le risque avec Airflow',
    schedule_interval='@daily',
)

def load_data(**kwargs):
    file_path = '/home/dhouha/airflow/data/clientscsvv.csv'
    df = pd.read_csv(file_path, delimiter=';')
    kwargs['ti'].xcom_push(key='data', value=df.to_dict())

def preprocess_data_task(**kwargs):
    data = kwargs['ti'].xcom_pull(task_ids='load_data_task', key='data')
    data = pd.DataFrame.from_dict(data)
    target_column = 'riskLevel (Y)'
    X_preprocessed, y_preprocessed, le_y, pipeline, preprocessor = preprocess_data(data, target_column)
    
    # Sauvegarde du pipeline
    os.makedirs(os.path.dirname(PIPELINE_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, PIPELINE_SAVE_PATH)
    
    le_y_classes = le_y.classes_.tolist()
    
    kwargs['ti'].xcom_push(key='X_preprocessed', value=X_preprocessed.tolist())
    kwargs['ti'].xcom_push(key='y_preprocessed', value=y_preprocessed.tolist())
    kwargs['ti'].xcom_push(key='le_y_classes', value=le_y_classes)

def train_model_task(**kwargs):
    X_preprocessed = kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='X_preprocessed')
    y_preprocessed = kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='y_preprocessed')
    le_y_classes = kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='le_y_classes')
    
    X_preprocessed = np.array(X_preprocessed)
    y_preprocessed = np.array(y_preprocessed)
    
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_preprocessed, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    base_models = [
        ('gb', GradientBoostingClassifier(learning_rate=0.1, max_depth=5, min_samples_leaf=1, min_samples_split=5, n_estimators=100, random_state=42)),
        ('rf', RandomForestClassifier(bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=42))
    ]

    meta_model = SVC(kernel='linear', probability=True, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    stacking_clf.fit(X_resampled, y_resampled)
    
    model_path = '/home/dhouha/airflow/models/stacking_clf.joblib'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(stacking_clf, model_path)
    
    kwargs['ti'].xcom_push(key='model_path', value=model_path)

def evaluate_model_task(**kwargs):
    model_path = kwargs['ti'].xcom_pull(task_ids='train_model_task', key='model_path')
    model = joblib.load(model_path)
    X_preprocessed = kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='X_preprocessed')
    y_preprocessed = kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='y_preprocessed')
    
    X_preprocessed = np.array(X_preprocessed)
    y_preprocessed = np.array(y_preprocessed)
    
    X_test = X_preprocessed
    y_test = y_preprocessed
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    classification_rep = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Classification Report:\n{classification_rep}")

def risk_coefficient_per_column_task(**kwargs):
    model = joblib.load('/home/dhouha/airflow/models/stacking_clf.joblib')
    
    X_preprocessed_df = pd.DataFrame(kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='X_preprocessed'))
    y_preprocessed_df = pd.Series(kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='y_preprocessed'))
    data_df = pd.read_csv('/home/dhouha/airflow/data/clientscsvv.csv', delimiter=';')

    feature_name = Variable.get('feature_name', default_var='Produits')
    feature_value = Variable.get('feature_value', default_var='Epargne')

    risk_percentage, risk_suggestion = calculate_risk_coefficient(
        model,
        X_preprocessed_df,
        y_preprocessed_df,
        data_df,
        feature_name,
        feature_value
    )

    print(f"Coefficient de risque pour {feature_name} = {feature_value}: {risk_percentage:.2f}%")
    print(f"Suggestion de risque associée : {risk_suggestion}")


from airflow.models import Variable

def risk_coefficient_score_task(**kwargs):
    model = joblib.load('/home/dhouha/airflow/models/stacking_clf.joblib')
    preprocess_pipeline = joblib.load(PIPELINE_SAVE_PATH)
    
    X_preprocessed_df = pd.DataFrame(kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='X_preprocessed'))
    y_preprocessed_df = pd.Series(kwargs['ti'].xcom_pull(task_ids='preprocess_data_task', key='y_preprocessed'))
    data_df = pd.read_csv('/home/dhouha/airflow/data/clientscsvv.csv', delimiter=';')

    feature_dict = {
        'age': 64,
        'nationality': 'Cameroun',
        'gender': 'femme',
        'Activites_label': 'INDUSTRIES INFORMATIQUES ',
        'Produits': 'SANTE',
        'Relation': 'père',
        'Pays': 'FRANCE',
        'isPEP': False,
        'famCode': 'M',
        'VoieDeDistribution': 'Bureaux directs'
    }

    predicted_risk, risk_percentages, risk_coefficient = predict_risk_score(
        model,
        preprocess_pipeline,
        feature_dict
    )

    print(f"La classe prédite est: {predicted_risk}")
    print("Probabilités pour chaque classe:")
    for risk, percentage in risk_percentages.items():
        print(f"{risk}: {percentage:.2f}%")
    print(f"Le score de risque global est: {risk_coefficient:.2f}%")

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data_task',
    python_callable=preprocess_data_task,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_model_task,
    provide_context=True,
    dag=dag,
)

risk_coefficient_per_column_task = PythonOperator(
    task_id='risk_coefficient_per_column_task',
    python_callable=risk_coefficient_per_column_task,
    provide_context=True,
    dag=dag,
)

risk_coefficient_score_task = PythonOperator(
    task_id='risk_coefficient_score_task',
    python_callable=risk_coefficient_score_task,
    provide_context=True,
    dag=dag,
)

load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task
evaluate_model_task >> risk_coefficient_per_column_task
evaluate_model_task >> risk_coefficient_score_task
