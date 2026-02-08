import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi, login

def load_data_from_hf():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    try:
        file_path = hf_hub_download(
            repo_id="tourism-package-prediction-data",
            filename="tourism.csv",
            repo_type="dataset"
        )
        df = pd.read_csv(file_path)
        print(f"Dataset loaded from Hugging Face, Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    df_clean = df.copy()
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop('Unnamed: 0', axis=1)
    if 'CustomerID' in df_clean.columns:
        df_clean = df_clean.drop('CustomerID', axis=1)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    print(f"Data cleaned, Final shape: {df_clean.shape}")
    return df_clean

def encode_categorical_features(df):
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}': {len(le.classes_)} unique values")
    return df_encoded, label_encoders

def split_and_save_data(df):
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    os.makedirs("tourism_project/data", exist_ok=True)
    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    return train_path, test_path

def upload_to_hf(train_path, test_path):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    api = HfApi()
    repo_id = "tourism-package-prediction-data"
    try:
        api.upload_file(path_or_fileobj=train_path, path_in_repo="train.csv", repo_id=repo_id, repo_type="dataset")
        print(f"Train data uploaded to {repo_id}")
        api.upload_file(path_or_fileobj=test_path, path_in_repo="test.csv", repo_id=repo_id, repo_type="dataset")
        print(f"Test data uploaded to {repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        raise

def main():
    print("DATA PREPARATION PIPELINE")
    df = load_data_from_hf()
    df_clean = clean_data(df)
    df_encoded, label_encoders = encode_categorical_features(df_clean)
    train_path, test_path = split_and_save_data(df_encoded)
    upload_to_hf(train_path, test_path)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
