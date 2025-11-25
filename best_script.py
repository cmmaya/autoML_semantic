from sklearn.linear_model._logistic import LogisticRegressionCV

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from typing import Dict, Any
from examples.contract import Optimizable, DataFrame, Hyperparameters, Metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class TitanicLogRegBaseline(Optimizable):
    """An optimizable script for the Titanic dataset."""
    DEFAULT_HPARAMS: Hyperparameters = {'model__Cs': 10, 'model__fit_intercept': True, 'model__cv': None, 'model__dual': False, 'model__penalty': 'l2', 'model__scoring': None, 'model__solver': 'lbfgs', 'model__tol': 0.0001, 'model__max_iter': 100, 'model__class_weight': None, 'model__n_jobs': None, 'model__verbose': 0, 'model__refit': True, 'model__intercept_scaling': 1.0, 'model__multi_class': 'deprecated', 'model__random_state': None, 'model__l1_ratios': None}

    def run(self, hparams: Hyperparameters) -> Metrics:
        """Runs the training and evaluation pipeline."""
        df = self.data.copy()
        y = df['Survived'].astype(int)
        X = df.drop(columns=['Survived'])
        numeric_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        categorical_features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=hparams.get('test_size', 0.2), stratify=y, random_state=hparams.get('random_state', 42))
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preprocess = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
        model = LogisticRegressionCV()
        pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
        if hparams.get('use_grid_search', True):
            param_grid = {}
            search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=hparams.get('scoring', 'roc_auc'), cv=hparams.get('cv', 5), n_jobs=1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
        else:
            pipeline.set_params(**{'model__C': hparams.get('C', 1.0), 'model__penalty': hparams.get('penalty', 'l2'), 'model__class_weight': hparams.get('class_weight', None)})
            pipeline.fit(X_train, y_train)
            best = pipeline
        if hasattr(best.named_steps['model'], 'predict_proba'):
            scores = best.predict_proba(X_test)[:, 1]
        else:
            scores = best.decision_function(X_test)
        auc = float(roc_auc_score(y_test, scores))
        return {'auc': auc}
if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    script = TitanicLogRegBaseline(data)
    baseline_metrics = script.run(hparams=TitanicLogRegBaseline.DEFAULT_HPARAMS)
    print(f'Baseline Run Metrics: {baseline_metrics}')