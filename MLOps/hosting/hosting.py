from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="MLOps/deployment",
    repo_id="RStoopAi/PIMA_Diabetes_Prediction",                                         
    repo_type="space",
    path_in_repo="",                          
)
