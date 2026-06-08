import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

mlflow.set_tracking_uri('http://127.0.0.1:8000/')
mlflow.set_experiment("Eksperimen_Sleep_Disorder")

df = pd.read_csv("data_preprocessing.csv")
X = df.drop(columns=['Sleep Disorder'])
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name='RF_Basic'):
    n_estimators = 500
    max_depth = 40

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
