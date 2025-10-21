# automl_lib/knowledge/ingestion/prompts.py
"""
Contains master prompts for the LLM-powered ingestion parser.
"""

# This prompt is engineered to force the LLM to return structured JSON.
COMPONENT_ANALYSIS_PROMPT = """
You are an expert Machine Learning engineer creating a knowledge graph. Analyze the provided documentation for a single Python class or function. Based ONLY on the text, extract the following information in a valid JSON format.

**JSON Schema:**
{
  "uid": "string (the full import path, e.g., 'torch.nn.Linear')",
  "name": "string (the class name, e.g., 'Linear')",
  "component_type": "string (one of: 'Layer', 'Model', 'Activation', 'Loss', 'Optimizer', 'Transformer', 'Classifier', 'Regressor', 'TimeSeriesModel')",
  "hyperparameters": [
    {
      "name": "string",
      "data_type": "string (e.g., 'int', 'float', 'bool', 'str', 'tuple')",
      "description": "string (a brief summary)"
    }
  ],
  "io_info": {
    "accepts_input": "string (describe the expected input, e.g., 'Tensor of shape [N, *, in_features]')",
    "produces_output": "string (describe the output, e.g., 'Tensor of shape [N, *, out_features]')"
  }
}

**Rules:**
1.  Adhere strictly to the JSON schema.
2.  Infer the `component_type` based on the component's primary function.
3.  For `hyperparameters`, only extract key constructor arguments.
4.  If information is not present, use an empty list or an empty string.

**Documentation to Analyze:**
---
{documentation_text}
---
"""