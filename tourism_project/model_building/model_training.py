import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import hf_hub_download, HfApi, login
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_train_test_data():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    try:
        train_path = hf_hub_download(repo_id="tourism-package-prediction-data", filename="train.csv", repo_type="dataset")
        train_df = pd.read_csv(train_path)
        test_path = hf_hub_download(repo_id="tourism-package-prediction-data", filename="test.csv", repo_type="dataset")
        test_df = pd.read_csv(test_path)
        X_train = train_df.drop('ProdTaken', axis=1)
        y_train = train_df['ProdTaken']
        X_test = test_df.drop('ProdTaken', axis=1)
        y_test = test_df['ProdTaken']
        print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    return metrics

def train_and_log_model(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        return model, metrics

def train_all_models(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Tourism_Package_Prediction")
    results = {}

    dt_params = {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10, 'random_state': 42}
    dt_model, dt_metrics = train_and_log_model(DecisionTreeClassifier(**dt_params), "Decision_Tree", dt_params, X_train, X_test, y_train, y_test)
    results['Decision_Tree'] = {'model': dt_model, 'metrics': dt_metrics}

    rf_params = {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1}
    rf_model, rf_metrics = train_and_log_model(RandomForestClassifier(**rf_params), "Random_Forest", rf_params, X_train, X_test, y_train, y_test)
    results['Random_Forest'] = {'model': rf_model, 'metrics': rf_metrics}

    ada_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
    ada_model, ada_metrics = train_and_log_model(AdaBoostClassifier(**ada_params), "AdaBoost", ada_params, X_train, X_test, y_train, y_test)
    results['AdaBoost'] = {'model': ada_model, 'metrics': ada_metrics}

    gb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'random_state': 42}
    gb_model, gb_metrics = train_and_log_model(GradientBoostingClassifier(**gb_params), "Gradient_Boosting", gb_params, X_train, X_test, y_train, y_test)
    results['Gradient_Boosting'] = {'model': gb_model, 'metrics': gb_metrics}

    xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'}
    xgb_model, xgb_metrics = train_and_log_model(XGBClassifier(**xgb_params), "XGBoost", xgb_params, X_train, X_test, y_train, y_test)
    results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}

    return results

def select_best_model(results):
    comparison = []
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison.append({'Model': model_name, 'F1-Score': metrics['f1_score']})
    comparison_df = pd.DataFrame(comparison).sort_values('F1-Score', ascending=False)
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    print(f"Best Model: {best_model_name}")
    return best_model, best_model_name

def save_and_upload_model(model, model_name):
    os.makedirs("tourism_project/model_building", exist_ok=True)
    model_path = f"tourism_project/model_building/{model_name}_model.pkl"
    joblib.dump(model, model_path)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    api = HfApi()
    repo_id = "tourism-package-prediction-model"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=model_path, path_in_repo=f"{model_name}_model.pkl", repo_id=repo_id, repo_type="model")
    print(f"Model uploaded to {repo_id}")

def main():
    print("MODEL TRAINING PIPELINE")
    X_train, X_test, y_train, y_test = load_train_test_data()
    results = train_all_models(X_train, X_test, y_train, y_test)
    best_model, best_model_name = select_best_model(results)
    save_and_upload_model(best_model, best_model_name)
    print("MODEL TRAINING COMPLETED")

if __name__ == "__main__":
    main()
