# ===============================================================
# retriever_agent.py
# Interfaz con GPT-5 para generar scripts candidatos
# ===============================================================
from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import requests
from rich.console import Console
from typing import List, Dict

console = Console()

GPT5_API_KEY = os.getenv("GPT5_API_KEY")
GPT5_API_URL = "https://api.openai.com/v1/chat/completions"
GPT5_MODEL = "gpt-5"  # Ajusta según la versión disponible

if not GPT5_API_KEY:
    console.print("[bold red]❌ No se encontró la variable de entorno GPT5_API_KEY.[/bold red]")
    console.print("Configura tu clave con: export GPT5_API_KEY='tu_clave_api'")
    raise SystemExit(1)


# ===============================================================
# Función de limpieza del JSON antes del parseo
# ===============================================================
def sanitize_result_text(raw_text: str) -> str:
    """
    Limpia el texto JSON devuelto por GPT para evitar escapes dañinos (\t, \r, etc.)
    antes de hacer json.loads().
    """
    if not raw_text:
        return raw_text

    original_len = len(raw_text)

    # 🔹 Escapar manualmente secuencias peligrosas
    raw_text = raw_text.replace("\\t", "\\\\t")  # evita que \t se convierta en tab real
    raw_text = raw_text.replace("\\r", "\\\\r")  # evita retornos de carro
    raw_text = raw_text.replace("\\y", "\\\\y")  # evita corrupción en '\y'

    # 🔹 Extraer solo el bloque JSON si viene con markdown o texto adicional
    match = re.search(r"\[.*\]|\{.*\}", raw_text, re.DOTALL)
    if match:
        raw_text = match.group(0)

    # 🔹 Quitar posibles comillas triples o marcas Markdown
    raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE)

    # 🔹 Limpieza general
    raw_text = raw_text.strip().replace("\u0000", "")

    cleaned_len = len(raw_text)
    console.print(f"[cyan]🧹 Limpieza aplicada: {original_len - cleaned_len} caracteres removidos/ajustados.[/cyan]")

    return raw_text


# ===============================================================
# Clase principal del agente Retriever
# ===============================================================
class RetrieverAgent:
    """
    Clase encargada de interactuar con GPT-5 para generar ejemplos de scripts
    adaptados al dataset analizado.
    """

    def __init__(self):
        self.api_key = GPT5_API_KEY
        self.model = GPT5_MODEL

    def _build_prompt(self, dataset_summary: Dict) -> str:
        prompt =f"""
You are an AutoML expert.

Generate 3 different script proposals in **valid JSON** format.
Each script must:
- Use well-known libraries (scikit-learn, XGBoost, or LightGBM)
- Define the following key variables explicitly:
  - `model`: the main estimator instance
  - `param_grid`: dictionary of hyperparameters to tune
  - `auc`: final float metric (Area Under ROC Curve)
- Train and evaluate the model on the given dataset
- Optimize for the metric: {dataset_summary.get('metric', 'auc')}
- Print ONLY a single JSON line at the end in the format:
  {{"auc": <numeric_value>}}

⚠️ Important:
- The script must end by printing ONLY this JSON line (no extra text, logs, or explanations).
- Ensure 'auc' is a float value (e.g., 0.8421).
- Avoid print statements or outputs that would break JSON parsing.
- Use clear variable names: model, param_grid, auc.
- Do not rename or nest these variables.
- Do not wrap the final print inside functions or classes.

Response format (must be strictly valid JSON, no markdown, no text outside the array):

[
  {{
    "name": "Model Name",
    "description": "Short one-line explanation",
    "code_snippet": "Minimal working Python code that prints {{\\"auc\\": value}}"
  }},
  {{
    "name": "...",
    "description": "...",
    "code_snippet": "..."
  }},
  {{
    "name": "...",
    "description": "...",
    "code_snippet": "..."
  }}
]

Do not exceed 7000 tokens.

Dataset summary:
{json.dumps(dataset_summary, indent=2)}
"""


        return prompt.strip()

    def retrieve_contextual_examples(self, dataset_summary: Dict) -> List[Dict]:
        """Realiza una única llamada a GPT-5 para obtener scripts candidatos."""
        prompt = self._build_prompt(dataset_summary)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "max_completion_tokens": 8000,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente experto en AutoML y pipelines de aprendizaje automático. "
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        console.print("[bold yellow]🧠 Llamando a GPT-5 para generar scripts...[/bold yellow]")
        response = requests.post(GPT5_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            console.print(f"[red]Error en la API GPT-5: {response.status_code}[/red]")
            console.print(response.text)
            return []

        result_text = response.json()["choices"][0]["message"]["content"]
        # result_text = '[\n  {\n    "name": "Sklearn Logistic AUC Pipeline",\n    "description": "Pipeline con OHE e imputación + GridSearchCV optimizando AUC.",\n    "code_snippet": "import pandas as pd\\nfrom sklearn.model_selection import StratifiedKFold, GridSearchCV\\nfrom sklearn.pipeline import Pipeline\\nfrom sklearn.compose import ColumnTransformer\\nfrom sklearn.preprocessing import OneHotEncoder\\nfrom sklearn.impute import SimpleImputer\\nfrom sklearn.linear_model import LogisticRegression\\n\\ndf=pd.read_csv(\\"data\\\\\\\\train.csv\\")\\ny=df[\\"Survived\\"]\\nX=df.drop(columns=[\\"Survived\\"])\\nnum=X.select_dtypes(\\"number\\").columns\\ncat=X.columns.difference(num)\\npre=ColumnTransformer([\\n    (\\"n\\",SimpleImputer(strategy=\\"median\\"),num),\\n    (\\"c\\",Pipeline([(\\"imp\\",SimpleImputer(strategy=\\"most_frequent\\")),(\\"ohe\\",OneHotEncoder(handle_unknown=\\"ignore\\"))]),cat)\\n])\\npipe=Pipeline([(\\"pre\\",pre),(\\"clf\\",LogisticRegression(max_iter=2000,class_weight=\\"balanced\\",solver=\\"liblinear\\"))])\\nparam={\\"clf__C\\":[0.1,0.3,1,3,10],\\"clf__penalty\\":[\\"l1\\",\\"l2\\"]}\\ncv=StratifiedKFold(5,shuffle=True,random_state=42)\\ngs=GridSearchCV(pipe,param,scoring=\\"roc_auc\\",cv=cv,n_jobs=-1)\\ngs.fit(X,y)\\nprint(\\"Best AUC:\\",gs.best_score_,\\"Params:\\",gs.best_params_)"\n  },\n  {\n    "name": "XGBoost AUC Pipeline",\n    "description": "XGBoost con preprocesamiento y RandomizedSearchCV maximizando AUC.",\n    "code_snippet": "import pandas as pd\\nfrom sklearn.model_selection import StratifiedKFold, RandomizedSearchCV\\nfrom sklearn.pipeline import Pipeline\\nfrom sklearn.compose import ColumnTransformer\\nfrom sklearn.preprocessing import OneHotEncoder\\nfrom sklearn.impute import SimpleImputer\\nimport xgboost as xgb\\n\\ndf=pd.read_csv(\\"data\\\\\\\\train.csv\\")\\ny=df[\\"Survived\\"]\\nX=df.drop(columns=[\\"Survived\\"])\\nnum=X.select_dtypes(\\"number\\").columns\\ncat=X.columns.difference(num)\\npre=ColumnTransformer([\\n    (\\"n\\",SimpleImputer(strategy=\\"median\\"),num),\\n    (\\"c\\",Pipeline([(\\"imp\\",SimpleImputer(strategy=\\"most_frequent\\")),(\\"ohe\\",OneHotEncoder(handle_unknown=\\"ignore\\"))]),cat)\\n])\\nspw=(len(y)-y.sum())/y.sum()\\nclf=xgb.XGBClassifier(objective=\\"binary:logistic\\",eval_metric=\\"auc\\",tree_method=\\"hist\\",n_estimators=800,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,scale_pos_weight=spw)\\npipe=Pipeline([(\\"pre\\",pre),(\\"clf\\",clf)])\\nparam={\\n \\"clf__max_depth\\":[3,4,5,6],\\n \\"clf__min_child_weight\\":[1,2,4,6],\\n \\"clf__gamma\\":[0,0.1,0.3],\\n \\"clf__reg_alpha\\":[0,0.1,0.5],\\n \\"clf__reg_lambda\\":[0.5,1,2]\\n}\\ncv=StratifiedKFold(5,shuffle=True,random_state=42)\\nrs=RandomizedSearchCV(pipe,param,n_iter=25,scoring=\\"roc_auc\\",cv=cv,n_jobs=-1,random_state=42)\\nrs.fit(X,y)\\nprint(\\"Best AUC:\\",rs.best_score_,\\"Params:\\",rs.best_params_)"\n  },\n  {\n    "name": "LightGBM AUC Pipeline",\n    "description": "LightGBM con categorías nativas y RandomizedSearchCV optimizando AUC.",\n    "code_snippet": "import pandas as pd\\nfrom sklearn.model_seleccion import StratifiedKFold, RandomizedSearchCV\\nimport lightgbm as lgb\\n\\ndf=pd.read_csv(\\"data\\\\\\\\train.csv\\")\\ny=df[\\"Survived\\"]\\nX=df.drop(columns=[\\"Survived\\"])\\ncat=X.select_dtypes(exclude=[\\"number\\"]).columns\\nfor c in cat: X[c]=X[c].astype(\\"category\\")\\nspw=(len(y)-y.sum())/y.sum()\\nclf=lgb.LGBMClassifier(objective=\\"binary\\",metric=\\"auc\\",n_estimators=1500,learning_rate=0.03,subsample=0.9,colsample_bytree=0.8,random_state=42,n_jobs=-1,scale_pos_weight=spw,verbose=-1)\\nparam={\\n \\"num_leaves\\":[31,63,127],\\n \\"max_depth\\":[-1,5,7,9],\\n \\"min_child_samples\\":[10,20,40],\\n \\"reg_alpha\\":[0,0.1,0.5],\\n \\"reg_lambda\\":[0,0.1,0.5]\\n}\\ncv=StratifiedKFold(5,shuffle=True,random_state=42)\\nrs=RandomizedSearchCV(clf,param,n_iter=20,scoring=\\"roc_auc\\",cv=cv,n_jobs=-1,random_state=42)\\nrs.fit(X,y)\\nprint(\\"Best AUC:\\",rs.best_score_,\\"Params:\\",rs.best_params_)"\n  }\n]'
        # 🔧 Sanitización antes del parseo JSON
        # ==========================================================
        clean_text = sanitize_result_text(result_text)

        # ==========================================================
        # Intento de parseo
        # ==========================================================
        try:
            examples = json.loads(clean_text)
            console.print(f"[green]✅ GPT-5 devolvió {len(examples)} candidatos.[/green]")
            return examples
        except json.JSONDecodeError as e:
            console.print(f"[red]⚠ No se pudo parsear el JSON devuelto por GPT-5.[/red]")
            console.print(f"[yellow]Error: {e}[/yellow]")
            console.print(f"[dim]{clean_text[:500]}...[/dim]")  # vista previa
            return []
