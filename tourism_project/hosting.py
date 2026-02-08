import os
from huggingface_hub import HfApi, login

def push_to_huggingface_space():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    else:
        print("Error: HF_TOKEN not found")
        return

    api = HfApi()
    space_id = "tourism-package-prediction-app"

    try:
        api.create_repo(repo_id=space_id, repo_type="space", space_sdk="streamlit", exist_ok=True, private=False)
        print(f"Space '{space_id}' created/verified")

        files = [
            ("tourism_project/deployment/Dockerfile", "Dockerfile"),
            ("tourism_project/deployment/app.py", "app.py"),
            ("tourism_project/deployment/requirements.txt", "requirements.txt")
        ]

        for local_path, repo_path in files:
            if os.path.exists(local_path):
                api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=space_id, repo_type="space")
                print(f"Uploaded {repo_path}")

        print(f"All files uploaded to {space_id}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    push_to_huggingface_space()
