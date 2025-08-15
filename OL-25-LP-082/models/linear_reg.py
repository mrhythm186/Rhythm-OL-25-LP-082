import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('cleaned_survey.csv')
features = ['Gender','self_employed','family_history','remote_work',
            'tech_company','benefits','care_options','wellness_program',
            'seek_help','anonymity','leave','mental_health_consequence',
            'phys_health_consequence','coworkers','supervisor',
            'mental_health_interview','phys_health_interview','mental_vs_physical',
            'obs_consequence','no_employees','work_interfere']
X = df[features]
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])
pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])
dummy_values = list(range(7))  
param_grid = {
    'model__fit_intercept': [True, False],
    'model__positive': [True, False],
    'model__copy_X': [True, False],
    'prep__cat__handle_unknown': ['ignore'] * 7  
}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
