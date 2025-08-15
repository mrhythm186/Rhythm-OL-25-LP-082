import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load dataset
df = pd.read_csv('cleaned_survey.csv')

# Feature columns and target
features = ['Gender', 'self_employed', 'family_history', 'work_interfere', 'remote_work',
            'benefits', 'care_options', 'wellness_program', 'seek_help', 'leave',
            'mental_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview']

X = df[features]
y = df['treatment'].map({'Yes': 1, 'No': 0})  # Encode target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = X.select_dtypes(include='object').columns.tolist()
preprocessing = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

pipe = Pipeline([
    ('pre', preprocessing),
    ('model', LogisticRegression(solver='saga', max_iter=10000, random_state=42))
])

param_grid = {
    'model__penalty': ['l1', 'l2', 'elasticnet', None],
    'model__C': [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0, 0.5, 1] 
}
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=10,
    cv=6,
    scoring='roc_auc',
    random_state=42
)
search.fit(X_train, y_train)
y_pred = search.predict(X_test)
y_proba = search.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
joblib.dump(search,'clf_model.pkl')
