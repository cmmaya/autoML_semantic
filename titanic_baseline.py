# titanic_baseline.py
"""
A simple, optimizable baseline script for the Titanic survivor prediction task.
"""
import pandas as pd
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# Make sure the contract.py file is in a place Python can find it
# For this example, let's assume it's in `automl_lib/examples/contract.py`
from examples.contract import Optimizable, DataFrame, Hyperparameters, Metrics

class TitanicClassifier(Optimizable):
    """An optimizable script for the Titanic dataset."""

    # These are the default hyperparameters the Analyzer will discover
    DEFAULT_HPARAMS: Hyperparameters = {
        'model__n_estimators': 3,
        'model__max_depth': 1,
        'model__min_samples_leaf': 1,
        'model__criterion': 'gini'
    }

    def run(self, hparams: Hyperparameters) -> Metrics:
        """Runs the training and evaluation pipeline."""
        df = self.data.copy()

        # 1. Define features and target
        target = 'Survived'
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        
        X = df[numerical_features + categorical_features]
        y = df[target]

        # 2. Create preprocessing pipelines for numerical and categorical data
        numerical_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 3. Create a preprocessor object using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # 4. Define the model
        model = RandomForestClassifier(random_state=42)

        # 5. Create the full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # 6. Set the hyperparameters for this run
        pipeline.set_params(**hparams)

        # 7. Split data and train
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipeline.fit(X_train, y_train)

        # 8. Evaluate
        preds_proba = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds_proba)
        
        return {'auc': auc}

if __name__ == '__main__':
    # This block allows us to run the script directly to get a baseline score
    data = pd.read_csv('data/train.csv')
    script = TitanicClassifier(data)
    baseline_metrics = script.run(hparams=TitanicClassifier.DEFAULT_HPARAMS)
    print(f"Baseline Run Metrics: {baseline_metrics}")