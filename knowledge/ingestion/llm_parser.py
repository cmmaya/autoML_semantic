# automl_lib/knowledge/ingestion/llm_parser.py
"""
A universal parser that uses an LLM to read documentation and build
a knowledge graph representation of a library.
"""

import json
import logging
from typing import Iterator, Tuple, List, Dict, Any

# Third-party libraries for web scraping. Add 'requests' and 'beautifulsoup4' to requirements.txt
import requests
from bs4 import BeautifulSoup

from .base_parser import BaseParser
from .prompts import COMPONENT_ANALYSIS_PROMPT
from knowledge.ontology import Node, ComponentNode, Relationship, DataTypeNode

def _call_llm_api(prompt: str) -> str:
    """A placeholder for a real LLM API call (e.g., to OpenAI, Anthropic, or a local model)."""
    # In a real implementation, this would make an HTTP request.
    #TODO: Integrate with an actual LLM service.
    # For this example, we'll return a hardcoded JSON string for a known input.
    logging.warning("Using placeholder LLM API. Only 'torch.nn.Linear' will be parsed correctly.")
    if "Linear" in prompt and "in_features" in prompt:
        return json.dumps({
            "uid": "torch.nn.Linear", "name": "Linear", "component_type": "Layer",
            "hyperparameters": [
                {"name": "in_features", "data_type": "int", "description": "Size of each input sample."},
                {"name": "out_features", "data_type": "int", "description": "Size of each output sample."},
                {"name": "bias", "data_type": "bool", "description": "If set to False, the layer will not learn an additive bias."}
            ],
            "io_info": {
                "accepts_input": "Tensor of shape [N, *, in_features]",
                "produces_output": "Tensor of shape [N, *, out_features]"
            }
        })
    return "{}" # Return empty JSON for unknown inputs

class LLMParser(BaseParser):
    """Uses an LLM to parse library documentation from URLs."""

    def __init__(self, targets: List[Dict[str, str]], concepts: List[Node]):
        """
        Args:
            targets: A list of dicts, each with a 'uid' and a 'url'.
            concepts: A list of abstract concept nodes for linking.
        """
        self.targets = targets
        self.concepts = {node.name: node for node in concepts}

    def _scrape_doc_text(self, url: str) -> str:
        """Scrapes the main content text from a given documentation URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # This selector is common for documentation sites (e.g., Sphinx themes)
            main_content = soup.find('div', role='main') or soup.find('article')
            return main_content.get_text(separator=' ', strip=True) if main_content else ""
        except requests.RequestException as e:
            logging.error(f"Failed to scrape URL {url}: {e}")
            return ""

    def parse(self) -> Iterator[Tuple[Node, list[Relationship]]]:
        """Iterates through targets, scrapes docs, and uses LLM to extract info."""
        logging.info(f"Starting LLM parsing for {len(self.targets)} targets...")

        for concept_node in self.concepts.values():
            yield concept_node, []

        for target in self.targets:
            uid, url = target['uid'], target['url']
            logging.info(f"Processing '{uid}' from '{url}'...")
            
            doc_text = self._scrape_doc_text(url)
            if not doc_text:
                continue

            prompt = COMPONENT_ANALYSIS_PROMPT.format(documentation_text=doc_text[:8000]) # Limit context window
            
            try:
                llm_response_str = _call_llm_api(prompt)
                llm_data = json.loads(llm_response_str)

                if not llm_data: continue

                node = ComponentNode(
                    uid=llm_data.get('uid'),
                    name=llm_data.get('name'),
                    component_type=llm_data.get('component_type'),
                    hyperparameters={h['name']: h for h in llm_data.get('hyperparameters', [])}
                )
                
                relationships = []
                # Attempt to link to a known concept
                comp_type = node.component_type + " Concept"
                if comp_type in self.concepts:
                    relationships.append(Relationship(node.uid, self.concepts[comp_type].uid, "is_a"))

                yield node, relationships

            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Failed to parse LLM response for {uid}: {e}")