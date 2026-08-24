from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="fraud_mlops_pipeline",
    start_date=datetime(2026, 8, 24),
    schedule=None,
    catchup=False,
    tags=["fraud-mlops"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/fraud_mlops/train.py"
        ),
    )