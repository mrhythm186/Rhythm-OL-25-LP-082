import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score, confusion_matrix
from xgboost import XGBClassifier
df = pd.read_csv("cleaned_survey.csv")
features = ['Age','Gender', 'self_employed', 'family_history', 'work_interfere', 
            'remote_work', 'benefits', 'care_options', 'wellness_program', 
            'seek_help', 'leave', 'mental_health_consequence', 'coworkers', 
            'supervisor', 'mental_health_interview']
X = df[features]
y = df['treatment'].map({"Yes":1, "No":0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def build_processing_pipeline(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('nums', num_trans, num_cols),
        ('cats', cat_trans, cat_cols)
    ])
    return preprocessor
pipe_xgb = Pipeline([
    ('preprocessor', build_processing_pipeline(X_train)),
    ('clf', XGBClassifier(eval_metric='logloss', random_state=42))
])
param_grid = {
    'clf__n_estimators':[100,150,200],
    'clf__max_depth':[3,5,7],
    'clf__learning_rate':[0.01,0.1],
    'clf__subsample':[0.8,1.0],
    'clf__colsample_bytree':[0.8,1.0]
}
gs_xgb = GridSearchCV(pipe_xgb, param_grid=param_grid, cv=6, n_jobs=-1, verbose=2)
gs_xgb.fit(X_train, y_train)
y_prob = gs_xgb.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.54).astype(int)  
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
