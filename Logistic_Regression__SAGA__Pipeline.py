import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv(r'data/train.csv')
y = df['Survived'].astype(int)
X = df.drop(columns=['Survived'])

# Columns
numeric_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Model and grid
model = LogisticRegression(solver='saga', max_iter=2000)
param_grid = {
    'model__C': [0.1, 1.0, 10.0],
    'model__penalty': ['l2', 'l1'],
    'model__class_weight': [None, 'balanced']
}

pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', model)])

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

search.fit(X_train, y_train)
best = search.best_estimator_

if hasattr(best.named_steps['model'], 'predict_proba'):
    scores = best.predict_proba(X_test)[:, 1]
else:
    scores = best.decision_function(X_test)

auc = float(roc_auc_score(y_test, scores))
print(json.dumps({"auc": auc}))
