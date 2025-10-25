import os, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json

# Load data
df = pd.read_csv(r'data/train.csv')
X = df.drop(columns=['Survived'])
y = df['Survived']

# Feature types
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Model and grid
model = LogisticRegression(max_iter=500, solver='liblinear')
param_grid = {
    'model__C': [0.1, 1.0, 10.0],
    'model__penalty': ['l2'],
    'model__solver': ['liblinear']
}

pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit and evaluate
grid.fit(X_train, y_train)
y_proba = grid.predict_proba(X_test)[:, 1]
auc = float(roc_auc_score(y_test, y_proba))

print(json.dumps({'auc': auc}))