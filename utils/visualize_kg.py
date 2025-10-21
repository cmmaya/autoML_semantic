import pickle
import logging
from pathlib import Path
import networkx as nx
from pyvis.network import Network

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_subgraph(
    graph_path: Path = Path("knowledge_graph.pkl"),
    filter_prefix: str = "sklearn",
    output_filename: str = "scikit_learn_kg.html"
):
    """
    Loads a knowledge graph, filters it, and creates a stable, interactive visualization.

    Args:
        graph_path: Path to the serialized knowledge graph file.
        filter_prefix: The UID prefix to filter for (e.g., "sklearn", "torch").
        output_filename: The name of the output HTML file.
    """
    if not graph_path.exists():
        logging.error(f"Knowledge graph file not found at '{graph_path}'. Please run populate_graph.py first.")
        return

    # 1. Load the complete knowledge graph
    try:
        with open(graph_path, "rb") as f:
            full_graph: nx.DiGraph = pickle.load(f)
        logging.info(f"Successfully loaded the full knowledge graph with {full_graph.number_of_nodes()} nodes.")
    except Exception as e:
        logging.error(f"Failed to load or parse the graph file: {e}")
        return

    # 2. Filter for the Scikit-learn subgraph
    subgraph = nx.DiGraph()
    for uid, data in full_graph.nodes(data=True):
        if uid.startswith(filter_prefix):
            subgraph.add_node(uid, **data)

    # Add edges connected to the filtered nodes
    for u, v, data in full_graph.edges(data=True):
        if subgraph.has_node(u) and subgraph.has_node(v):
            subgraph.add_edge(u, v, **data)
        elif subgraph.has_node(u) and not v.startswith(filter_prefix):
             subgraph.add_node(v, **full_graph.nodes[v])
             subgraph.add_edge(u, v, **data)

    logging.info(f"Created subgraph with {subgraph.number_of_nodes()} nodes for prefix '{filter_prefix}'.")

    # 3. Create an interactive pyvis network
    net = Network(height="900px", width="100%", notebook=False, directed=True)

    color_map = {
        "Classifier": "#42a5f5",  # Blue
        "Regressor": "#66bb6a",   # Green
        "Transformer": "#ffa726", # Orange
        "Concept": "#ec407a",     # Pink
    }
    default_color = "#bdbdbd" # Grey

    # 4. Add nodes and edges to the pyvis graph
    for uid, data in subgraph.nodes(data=True):
        component_type = data.get("component_type", "Concept")
        color = color_map.get(component_type, default_color)
        title = f"UID: {uid}\nType: {component_type}" # Tooltip on hover
        
        net.add_node(
            uid,
            label=data.get("name"),
            title=title,
            color=color,
            shape="dot",
            size=15 if component_type == "Concept" else 10
        )

    for u, v, data in subgraph.edges(data=True):
        net.add_edge(u, v, title=data.get("label"))

    # --- FIX STARTS HERE ---
    # 5. Configure the physics engine for stability
    # This tells the graph to run a layout algorithm (BarnesHut) for a limited
    # number of iterations and then turn off the physics, resulting in a static graph.
    physics_options = """
    const options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springConstant": 0.04,
          "springLength": 200
        },
        "stabilization": {
          "iterations": 1000
        }
      }
    }
    """
    net.set_options(physics_options)
    # --- FIX ENDS HERE ---

    # 6. Generate the HTML file
    try:
        net.save_graph(output_filename)
        logging.info(f"✅ Interactive graph saved to '{output_filename}'. Open this file in your browser.")
    except Exception as e:
        logging.error(f"Failed to save the visualization: {e}")


if __name__ == '__main__':
    visualize_subgraph()