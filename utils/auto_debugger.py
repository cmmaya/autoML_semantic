# ===============================================================
# utils/auto_debugger.py
# AutoDebugger: GPT-based single-attempt repair using raw API call
# ===============================================================

import os
import json
import requests
from pathlib import Path
from rich.console import Console

console = Console()


class AutoDebugger:
    def __init__(self, gpt_model: str = "gpt-4.1", max_loops: int = 1):
        """
        gpt_model: OpenAI model used for debugging (default: gpt-4.1)
        max_loops: max repair attempts per script (default: 1)
        """
        self.model = gpt_model
        self.max_loops = max_loops
        self.api_key = os.getenv("GPT5_API_KEY")
        self.api_url = os.getenv("GPT5_API_URL", "https://api.openai.com/v1/chat/completions")

        if not self.api_key:
            raise EnvironmentError("Missing GPT5_API_KEY environment variable.")

    # -----------------------------------------------------------
    def _call_api(self, prompt: str) -> str:
        """Calls the GPT API with the given prompt and returns text output."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a senior Python AutoML engineer."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"GPT API error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    # -----------------------------------------------------------
    def debug_script(self, script_path: Path, error_output: str) -> Path:
        """
        Attempts to fix a failing script using GPT.
        Overwrites the original file with the repaired code.
        Returns the same path.
        """
        console.print(f"[cyan]🛠 Debugging {script_path.name} using {self.model}...[/cyan]")
        script_code = script_path.read_text()

        for attempt in range(1, self.max_loops + 1):
            console.print(f"[yellow]🔁 Debug attempt {attempt}/{self.max_loops}[/yellow]")

            prompt = f"""
            You are an expert AutoML Python engineer.

            You will receive a Python script that failed execution. 
            Your task is to FIX ONLY the faulty block of code indicated by the error message — 
            do NOT rewrite or reformat the entire script.

            --- Script Start ---
            {script_code}
            --- Script End ---

            Error message:
            {error_output}

            Repair Guidelines:
            1. Modify ONLY the minimal block required to fix the error.
            2. Preserve:
            - The overall structure and indentation.
            - All variable names: 'model', 'param_grid', 'auc'.
            - Section order if comments exist (Data Loading, Model Definition, etc.).
            3. DO NOT:
            - Rename variables.
            - Change library imports.
            - Remove valid logic from unaffected sections.
            4. The fixed script must:
            - Be syntactically correct and runnable.
            - End by printing a single valid JSON object:
                print(json.dumps({{"auc": float(auc)}}))
            5. Maintain the script’s readability and indentation exactly.

            Output:
            - Return ONLY the complete fixed Python code (no explanations, markdown, or comments).
            """

            try:
                new_code = self._call_api(prompt)

                # ✅ Overwrite the same file instead of creating a new one
                script_path.write_text(new_code)

                console.print(f"[green]✅ Repaired script overwritten: {script_path.name}[/green]")
                return script_path

            except Exception as ex:
                console.print(f"[red]⚠ Debug attempt {attempt} failed: {ex}[/red]")

        console.print(f"[red]❌ All debugging attempts failed for {script_path.name}[/red]")
        return script_path
