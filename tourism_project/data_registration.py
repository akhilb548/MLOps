import os
from huggingface_hub import HfApi, login
import pandas as pd

def register_dataset():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    else:
        print("Warning: HF_TOKEN not found")
        return

    api = HfApi()
    repo_id = "tourism-package-prediction-data"
    repo_type = "dataset"

    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
        print(f"Repository '{repo_id}' created/verified")

        dataset_path = "tourism_project/data/tourism.csv"
        if os.path.exists(dataset_path):
            api.upload_file(
                path_or_fileobj=dataset_path,
                path_in_repo="tourism.csv",
                repo_id=repo_id,
                repo_type=repo_type
            )
            print(f"Dataset uploaded successfully to {repo_id}")
            df = pd.read_csv(dataset_path)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        else:
            print(f"Error: Dataset file not found at {dataset_path}")
    except Exception as e:
        print(f"Error during data registration: {str(e)}")
        raise

if __name__ == "__main__":
    register_dataset()
