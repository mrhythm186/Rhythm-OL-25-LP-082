import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('cleaned_survey.csv')

y = df['treatment'].map({'Yes': 1, 'No': 0})
features = ['Gender', 'self_employed', 'family_history', 'work_interfere', 
            'remote_work', 'benefits', 'care_options', 'wellness_program', 
            'seek_help', 'leave', 'mental_health_consequence', 'coworkers', 
            'supervisor', 'mental_health_interview']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
cat_cols = X.select_dtypes(include='object').columns.tolist()
prep = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)],
    remainder='passthrough'
)

rf_pipe = Pipeline([
    ('pre', prep),
    ('model', RandomForestClassifier(random_state=42, max_depth=1000))
])
param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [20, 50, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 'log2']
}
rand_search = RandomizedSearchCV(
    rf_pipe, param_distributions=param_dist,
    n_iter=10, cv=6, scoring='roc_auc', random_state=42
)
rand_search.fit(X_train, y_train)
y_pred = rand_search.predict(X_test)
print("=== Random Forest Classification ===")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
