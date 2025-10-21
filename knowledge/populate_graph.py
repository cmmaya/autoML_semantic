# automl_lib/knowledge/populate_graph.py
"""
The main, comprehensive script to populate the Knowledge Graph.
#TODO: Refactor into smaller modules if it grows too large.
#TODO: make fully agnostic to new libraries by using a config file.
This script serves as the master ingestor, orchestrating all available library
parsers (programmatic and LLM-based) to build a rich, unified knowledge base
of ML components. The final graph is then saved to a file for the optimizer to use.
"""
import logging
import pickle
from pathlib import Path

from .graph_db import KnowledgeGraph
from .ingestion.scikit_learn_parser import ScikitLearnParser
from .ingestion.llm_parser import LLMParser
from .ontology import Node

# --- Library Configurations for the LLMParser ---
# To extend, simply add a new configuration and instantiate an LLMParser for it.

PYTORCH_CONCEPTS = [
    Node(uid="concept.layer", name="Layer Concept"),
    Node(uid="concept.model", name="Model Concept"),
    Node(uid="concept.loss", name="Loss Concept"),
    Node(uid="concept.optimizer", name="Optimizer Concept"),
    Node(uid="concept.activation", name="Activation Concept"),
]
PYTORCH_TARGETS = [
    {"uid": "torch.nn.Linear", "url": "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"},
    {"uid": "torch.nn.Conv2d", "url": "https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"},
    {"uid": "torch.nn.ReLU", "url": "https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html"},
    {"uid": "torch.nn.Dropout", "url": "https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"},
    {"uid": "torch.optim.Adam", "url": "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html"},
    {"uid": "torch.nn.CrossEntropyLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"},
    {"uid": "torchvision.models.resnet18", "url": "https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html"},
]

TENSORFLOW_CONCEPTS = PYTORCH_CONCEPTS # Can reuse concepts
TENSORFLOW_TARGETS = [
    {"uid": "tensorflow.keras.layers.Dense", "url": "https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense"},
    {"uid": "tensorflow.keras.layers.Conv2D", "url": "https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D"},
    {"uid": "tensorflow.keras.layers.ReLU", "url": "https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU"},
    {"uid": "tensorflow.keras.optimizers.Adam", "url": "https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam"},
]

TIMESERIES_CONCEPTS = [ Node(uid="concept.timeseriesmodel", name="TimeSeriesModel Concept") ]
TIMESERIES_TARGETS = [
    {"uid": "prophet.Prophet", "url": "https://facebook.github.io/prophet/docs/quick_start.html"},
    {"uid": "statsmodels.tsa.arima.model.ARIMA", "url": "https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html"},
]


def populate(output_path: Path = Path("knowledge_graph.pkl")):
    """
    Runs all available ingestion parsers and builds the knowledge graph.

    Args:
        output_path: The path where the final serialized graph will be saved.
    """
    kg = KnowledgeGraph()
    
    # --- Instantiate all parsers ---
    # The list of parsers defines the scope of our knowledge graph.
    parsers = [
        # Programmatic parser for well-structured libraries
        ScikitLearnParser(),
        
        # LLM-based parsers for complex/compositional libraries
        # TODO: comment back in once LLM integration is done
        #LLMParser(targets=PYTORCH_TARGETS, concepts=PYTORCH_CONCEPTS),
        #LLMParser(targets=TENSORFLOW_TARGETS, concepts=TENSORFLOW_CONCEPTS),
        #LLMParser(targets=TIMESERIES_TARGETS, concepts=TIMESERIES_CONCEPTS),
    ]
    
    node_count = 0
    rel_count = 0
    
    # --- Run Ingestion Loop ---
    for parser in parsers:
        parser_name = parser.__class__.__name__
        # Add a more descriptive name for the LLMParser instances
        if isinstance(parser, LLMParser):
            parser_name += f" ({parser.targets[0]['uid'].split('.')[0]})"

        logging.info(f"--- Running Parser: {parser_name} ---")
        try:
            for node, relationships in parser.parse():
                if node and node.uid:
                    kg.add_node(node)
                    node_count += 1
                    for rel in relationships:
                        kg.add_relationship(rel)
                        rel_count += 1
        except Exception as e:
            logging.error(f"Parser {parser_name} failed with a critical error: {e}", exc_info=True)

    logging.info("--- Knowledge Graph Population Complete ---")
    logging.info(f"Total nodes added: {node_count}")
    logging.info(f"Total relationships added: {rel_count}")
    
    # --- Save the Final Graph ---
    try:
        with open(output_path, "wb") as f:
            pickle.dump(kg.graph, f)
        logging.info(f"✅ Knowledge graph successfully saved to '{output_path}'.")
    except IOError as e:
        logging.error(f"Failed to save knowledge graph to '{output_path}': {e}")
        return

    # --- Verification Step ---
    if node_count > 0:
        logging.info("--- Verifying Graph Content ---")
        # Query for a scikit-learn component
        sk_test_uid = "sklearn.ensemble.RandomForestClassifier"
        compatible_sk = kg.find_compatible_components(sk_test_uid)
        logging.info(f"Found {len(compatible_sk)} compatible classifiers for RandomForestClassifier.")
        
        # Query for a torch component (using the hardcoded placeholder from the LLM parser)
        torch_test_uid = "torch.nn.Linear"
        node_data = kg.graph.nodes.get(torch_test_uid)
        if node_data:
            logging.info(f"Successfully retrieved PyTorch component: '{node_data.get('name')}'")
        else:
            logging.warning(f"PyTorch component '{torch_test_uid}' not found in graph.")

if __name__ == '__main__':
    # Configure logging for detailed, professional output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    populate()