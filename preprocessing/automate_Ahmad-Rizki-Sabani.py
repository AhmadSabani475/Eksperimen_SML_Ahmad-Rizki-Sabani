from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump,load
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def preprocess_data(input_path, save_preprocessor_path, output_path):
    df = pd.read_csv(input_path)
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    df.drop(columns=['Person ID'], inplace=True)

    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = df['Systolic_BP'].astype(int)
    df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
    df.drop(columns=['Blood Pressure'], inplace=True)

    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    bmi_mapping = {'Normal' : 1, 'Overweight' : 2, 'Obese' : 3}
    df['BMI Category'] = df['BMI Category'].map(bmi_mapping)

    le_target = LabelEncoder()
    df['Sleep Disorder'] = le_target.fit_transform((df['Sleep Disorder']))

    X = df.drop(columns='Sleep Disorder')
    y = df['Sleep Disorder']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    category_features = ['Gender', 'Occupation']

    numeric_tranformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('encode', OrdinalEncoder())])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_tranformer, numeric_features),
        ('cat', categorical_transformer, category_features)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    dump(preprocessor, save_preprocessor_path)

    feature_names = numeric_features + category_features
    df_result = pd.DataFrame(X_train, columns=feature_names)
    df_result['Sleep Disorder'] = y_train.values
    df_result.to_csv(output_path, index=False)
    print(f"Data preprocessed disimpan di: {output_path}")

    df_header = pd.DataFrame(columns=feature_names)
    df_header.to_csv('data.csv', index=False)

    return X_train, X_test, y_train, y_test

def inference(new_data, load_path):
    preprocessor = load(load_path)
    tranformed_data = preprocessor.transform(new_data)
    return tranformed_data

if __name__ == "__main__":
    pipeline_path = 'preprocessor_pipeline.joblib'
    X_train, X_test, y_train, y_test = preprocess_data('../Sleep_health_and_lifestyle_dataset.csv', pipeline_path, 'data_preprocessing.csv')

    col = pd.read_csv('data_preprocessing.csv')
    feature_cols = [c for c in col.columns if c != 'Sleep Disorder']

    new_data = [27, 6.1, 7, 75, 3, 1, 70, 8000, 125, 82, 'Male', 'Nurse']
    new_data_df = pd.DataFrame([new_data], columns=feature_cols)

    numeric_cols = [c for c in feature_cols if c not in ['Gender', 'Occupation']]
    new_data_df[numeric_cols] = new_data_df[numeric_cols].astype(float)

    transformed_data = inference(new_data_df, pipeline_path)
    print("Hasil inference:", transformed_data)