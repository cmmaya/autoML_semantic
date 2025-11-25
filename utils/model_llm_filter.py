# utils/model_llm_filter.py

import os
import json
import logging
from typing import List, Dict, Any

import requests

from knowledge.graph_db import ComponentNode  # Ajusta si la ruta difiere

GPT5_API_KEY = os.getenv("GPT5_API_KEY")
GPT5_API_URL = "https://api.openai.com/v1/chat/completions"
GPT5_MODEL = "gpt-5"  # Ajustar si usas otra variante


def llm_filter_models(
    models: List[ComponentNode],
    dataset_summary: Dict[str, Any],
    top_k: int = 5
) -> List[ComponentNode]:
    """
    Usa GPT-5 para priorizar los modelos candidatos del Knowledge Graph
    según el resumen del dataset. Devuelve como máximo top_k modelos.

    Si no hay API key o hay error, devuelve la lista original.
    """
    if not models:
        return models

    if not GPT5_API_KEY:
        logging.warning("llm_filter_models: GPT5_API_KEY no definido. Se omite filtrado LLM.")
        return models

    # Serializar modelos de forma ligera para el prompt
    serialized_models = []
    for m in models:
        serialized_models.append({
            "uid": getattr(m, "uid", None),
            "name": getattr(m, "name", None),
            "component_type": getattr(m, "component_type", None),
            "library": getattr(m, "library", None),
        })

    try:
        ds_json = json.dumps(dataset_summary, indent=2, default=str)
        models_json = json.dumps(serialized_models, indent=2, default=str)
    except Exception as e:
        logging.error(f"llm_filter_models: error serializando datos para el prompt: {e}")
        return models

    # Construir prompt

    prompt = (
        f"Eres un asistente experto en AutoML y selección de modelos.\n\n"
        f"Se te proporciona:\n"
        f"1) Un resumen del dataset (tamaño, tipos de columnas, tipo de tarea, métrica objetivo).\n"
        f"2) Una lista de modelos candidatos del Knowledge Graph.\n\n"
        f"Objetivo: seleccionar como máximo {top_k} modelos que sean más adecuados para este dataset y esta tarea.\n"
        f"Prioriza modelos que:\n"
        f"- Sean apropiados para el tipo de tarea (clasificación vs regresión).\n"
        f"- Soporten la métrica indicada cuando aplique (por ejemplo, AUC en clasificación binaria).\n"
        f"- Sean robustos y razonables como baseline mejorados (por ejemplo, RandomForest, GradientBoosting, XGBoost, etc.).\n\n"
        f"IMPORTANTE: responde únicamente un JSON válido con la forma:\n"
        f'{{"selected_uids": ["uid_1", "uid_2", ...]}}\n'
        f"sin ningún texto adicional.\n\n"
        f"Resumen del dataset:\n{ds_json}\n\n"
        f"Modelos candidatos:\n{models_json}\n"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT5_API_KEY}"
    }

    payload = {
        "model": GPT5_MODEL,
        "max_completion_tokens": 8000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en AutoML y pipelines de aprendizaje automático. "
                    "Tu tarea es seleccionar modelos adecuados del Knowledge Graph."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        logging.info("llm_filter_models: llamando a GPT-5 para filtrar modelos.")
        response = requests.post(GPT5_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"llm_filter_models: fallo en la llamada a GPT-5: {e}")
        return models

    # Parsear salida
    try:
        parsed = json.loads(content)
        selected_uids = parsed.get("selected_uids", [])
        if not isinstance(selected_uids, list):
            logging.warning("llm_filter_models: 'selected_uids' no es una lista. Se omite filtrado LLM.")
            return models
    except Exception as e:
        logging.error(f"llm_filter_models: error parseando la respuesta de GPT-5: {e}")
        return models

    uid_to_model = {getattr(m, "uid", None): m for m in models}
    filtered = []

    for uid in selected_uids:
        if uid in uid_to_model:
            filtered.append(uid_to_model[uid])

    if not filtered:
        logging.warning("llm_filter_models: GPT-5 no devolvió uids válidos. Se mantiene lista original.")
        return models

    if len(filtered) > top_k:
        filtered = filtered[:top_k]

    return filtered
