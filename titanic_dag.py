import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
from train_model import train

def download_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df.to_csv('/home/xronixle/airflow_titanic/raw_titanic.csv', index=False)
    print(f"Загружено данных: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    return df.shape[0]

def clean_data():
    df = pd.read_csv('/home/xronixle/airflow_titanic/raw_titanic.csv')
    print(f"Начальный размер: {df.shape}")
    
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {initial_len - len(df)}")
    
    print(f"Распределение целевой переменной:\n{df['Survived'].value_counts()}")
    
    df.to_csv('/home/xronixle/airflow_titanic/df_clear.csv', index=False)
    print(f"Финальный размер: {df.shape}")
    return len(df)

def save_artifacts():
    report = f"""
===================================
Titanic ML Pipeline Report
Timestamp: {datetime.now()}
===================================
Pipeline completed successfully!
"""
    with open("/home/xronixle/airflow_titanic/pipeline_report.txt", "w") as f:
        f.write(report)
    print(report)

default_args = {
    'owner': 'student',
    'start_date': datetime(2026, 3, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'titanic_pipeline',
    default_args=default_args,
    description='ML pipeline для Titanic',
    schedule_interval='@daily',
    catchup=False,
)

download_task = PythonOperator(task_id='download_data', python_callable=download_data, dag=dag)
clean_task = PythonOperator(task_id='clean_data', python_callable=clean_data, dag=dag)
train_task = PythonOperator(task_id='train_model', python_callable=train, dag=dag)
save_task = PythonOperator(task_id='save_artifacts', python_callable=save_artifacts, dag=dag)

download_task >> clean_task >> train_task >> save_task
