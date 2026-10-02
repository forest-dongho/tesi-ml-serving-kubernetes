
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kfp import dsl, compiler, kubernetes
from kfp.dsl import Dataset, Input, Model, Output

from components.data import prepare_data, data_train_test_split
from components.evaluation import model_evaluation
from components.export import export_model_for_serving


@dsl.component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def training(
    X_train_in: Input[Dataset],
    Y_train_in: Input[Dataset],
    model_out: Output[Model],
    scaler_out: Output[Model],
    max_iter: int = 10000,
):
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    with open(X_train_in.path, "rb") as f:
        X_train_data = pickle.load(f)
    with open(Y_train_in.path, "rb") as f:
        Y_train = pickle.load(f)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_data)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, Y_train)

    with open(model_out.path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_out.path, "wb") as f:
        pickle.dump(scaler, f)


@dsl.pipeline(
    name="breast-cancer-pipeline-logistic-regression",
    description="Breast Cancer Classification pipeline - Logistic Regression ",
)
def bc_pipeline_lr(
    test_size: float = 0.3,
    max_iter: int = 10000,
    s3_endpoint: str = "http://seaweedfs.kubeflow:9000",
    s3_bucket: str = "mlpipeline",
    s3_access_key: str = "",
    s3_secret_key: str = "",
):
    MODEL_TIER = "logistic-regression"

    prepared_data = prepare_data()
    kubernetes.add_pod_label(prepared_data, "model-tier", MODEL_TIER)
    kubernetes.add_pod_label(prepared_data, "stage", "prepare-data")

    split_data = data_train_test_split(
        features_in=prepared_data.outputs["features_out"],
        targets_in=prepared_data.outputs["targets_out"],
        test_size=test_size,
    )
    kubernetes.add_pod_label(split_data, "model-tier", MODEL_TIER)
    kubernetes.add_pod_label(split_data, "stage", "split")

    train_task = training(
        X_train_in=split_data.outputs["X_train_out"],
        Y_train_in=split_data.outputs["Y_train_out"],
        max_iter=max_iter,
    )
    kubernetes.add_pod_label(train_task, "model-tier", MODEL_TIER)
    kubernetes.add_pod_label(train_task, "stage", "training")


    model_task = model_evaluation(
        data_in=prepared_data.outputs["data_out"],
        model_in=train_task.outputs["model_out"],
        scaler_in=train_task.outputs["scaler_out"],
        X_test_in=split_data.outputs["X_test_out"],
        Y_test_in=split_data.outputs["Y_test_out"],
    )
    kubernetes.add_pod_label(model_task, "model-tier", MODEL_TIER)
    kubernetes.add_pod_label(model_task, "stage", "evaluation")

    export_task = export_model_for_serving(
        model_in=train_task.outputs["model_out"],
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_key=f"{MODEL_TIER}/model/model.joblib",
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    kubernetes.add_pod_label(export_task, "model-tier", MODEL_TIER)
    kubernetes.add_pod_label(export_task, "stage", "export")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=bc_pipeline_lr,
        package_path="pipeline_logistic_regression.yaml",
    )
    print("Pipeline compilata: pipeline_logistic_regression.yaml")
