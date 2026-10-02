
import sys
import os
import time
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_pipeline import run_and_wait
from pipelines.pipeline_logistic_regression import bc_pipeline_lr
from pipelines.pipeline_random_forest import bc_pipeline_rf
from pipelines.pipeline_gradient_boosting import bc_pipeline_gb


S3_ENDPOINT = "http://seaweedfs.kubeflow:9000"  
S3_BUCKET  = "mlpipeline"                      
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minio123")
# ==============================================================================

def common_s3_args():
    return {
        "s3_endpoint": S3_ENDPOINT,
        "s3_bucket": S3_BUCKET,
        "s3_access_key": S3_ACCESS_KEY,
        "s3_secret_key": S3_SECRET_KEY,
    }


def main():
    if not S3_ACCESS_KEY or not S3_SECRET_KEY:
        print("S3_ACCESS_KEY / S3_SECRET_KEY non impostate.")
        return


    execution_data = []
    
    print("=== 1/3: Logistic Regression (baseline) ===")
    start_time = time.time()
    run_and_wait(
        pipeline_func=bc_pipeline_lr,
        package_path="pipeline_logistic_regression.yaml",
        run_name="baseline-logistic-regression",
        arguments={**common_s3_args()},
    )
    duration_lr = time.time() - start_time
    execution_data.append(["Logistic Regression", duration_lr])
    print(f"✅ Completata in {duration_lr:.2f} secondi.\n")

    print("=== 2/3: Random Forest (baseline) ===")
    start_time = time.time()
    run_and_wait(
        pipeline_func=bc_pipeline_rf,
        package_path="pipeline_random_forest.yaml",
        run_name="baseline-random-forest",
        arguments={**common_s3_args()},
    )
    duration_rf = time.time() - start_time
    execution_data.append(["Random Forest", duration_rf])
    print(f"✅ Completata in {duration_rf:.2f} secondi.\n")

    print("=== 3/3: Gradient Boosting (baseline) ===")
    start_time = time.time()
    run_and_wait(
        pipeline_func=bc_pipeline_gb,
        package_path="pipeline_gradient_boosting.yaml",
        run_name="baseline-gradient-boosting",
        arguments={**common_s3_args()},
        timeout=3600,  
    )
    duration_gb = time.time() - start_time
    execution_data.append(["Gradient Boosting", duration_gb])
    print(f"✅ Completata in {duration_gb:.2f} secondi.\n")

    print("Tutte e tre le pipeline di baseline completate con successo.")
    print(f"Modelli disponibili su s3://{S3_BUCKET}/<model-tier>/model/model.joblib")

  
    csv_filename = "baseline_execution_times.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
       
        writer.writerow(["Pipeline", "Duration (Seconds)", "Duration (Minutes)"])
       
        for row in execution_data:
            pipeline_name = row[0]
            dur_sec = row[1]
            dur_min = dur_sec / 60.0
            writer.writerow([pipeline_name, f"{dur_sec:.2f}", f"{dur_min:.2f}"])
            
    print(f"\n Dati di esecuzione salvati con successo in: {csv_filename}")


if __name__ == "__main__":
    main()
