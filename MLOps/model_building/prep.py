
import pandas as pd
import sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/RStoopAi/PIMA_Diabetes_Prediction/pima.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

target_col = 'class'

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="RStoopAi/PIMA_Diabetes_Prediction",
        repo_type="dataset",
    )
