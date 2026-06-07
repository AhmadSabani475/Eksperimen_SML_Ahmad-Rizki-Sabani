import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss, classification_report,confusion_matrix, ConfusionMatrixDisplay
import dagshub
import matplotlib.pyplot as plt


# mlflow.set_tracking_uri("http://127.0.0.1:8000/")
# mlflow.set_experiment("Ekperiment_modelling_tuning")

dagshub.init(repo_owner='AhmadSabani475', repo_name='Eksperimen_SML_Ahmad-Rizki-Sabani', mlflow=True)
mlflow.set_experiment("Eksperimen_modelling_tuning")
df = pd.read_csv('data_preprocessing.csv')
X = df.drop(columns=['Sleep Disorder'])
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators' : [100, 200, 500],
    'max_depth' : [10, 20, 40],
    'min_samples_split' : [2, 5, 10]
}

with mlflow.start_run():
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    mlflow.log_metric("log_loss", log_loss(y_test, y_pred_proba))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    feature_importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")


    input_example =  X_train[0:5]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example
    )