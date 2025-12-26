
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

import joblib

import os

from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/RStoopAi/PIMA_Diabetes_Prediction/Xtrain.csv"                   
Xtest_path = "hf://datasets/RStoopAi/PIMA_Diabetes_Prediction/Xtest.csv"                  
ytrain_path = "hf://datasets/RStoopAi/PIMA_Diabetes_Prediction/ytrain.csv"                  
ytest_path = "hf://datasets/RStoopAi/PIMA_Diabetes_Prediction/ytest.csv"                      

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = [
    'preg',
    'plas',
    'pres',
    'skin',
    'test',
    'mass',
    'pedi',
    'age'
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

gb_model = GradientBoostingClassifier(random_state=42)

param_grid = {
    'gradientboostingclassifier__n_estimators': [75, 100, 125],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__subsample': [0.5, 0.6]
}

model_pipeline = make_pipeline(preprocessor, gb_model)

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)


best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

y_pred_train = best_model.predict(Xtrain)

y_pred_test = best_model.predict(Xtest)

print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

joblib.dump(best_model, "best_pima_diabetes_model_v1.joblib")

repo_id = "RStoopAi/PIMA_Diabetes_Prediction"                                        
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_pima_diabetes_model_v1.joblib",
    path_in_repo="best_pima_diabetes_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
