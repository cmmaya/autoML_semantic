

### 🏛️ Core Design Philosophy

The system is designed as an **intelligent code transformation engine**. It's not just a hyperparameter tuner; it's an automated programmer that reads, understands, modifies, and evaluates ML scripts to find a better solution.

1.  **Script as the Source of Truth**: The user's Python script is the complete definition of the problem. There are no registries or required configuration files.
2.  **Introspection over Registration**: The system discovers what to optimize—hyperparameters, architectures, pipeline steps—by analyzing the script's source code using Abstract Syntax Trees (AST) and Large Language Models (LLMs).
3.  **Knowledge-Driven Mutations**: A central **Knowledge Graph (KG)**, built with a foundational ontology and populated on-the-fly, provides the "wisdom" to make intelligent, semantically valid mutations.
4.  **Sandboxed, Black-Box Execution**: All user code is executed in a safe, isolated environment, ensuring stability and security. The optimizer only interacts with the script's inputs and outputs, not its internal state.

-----

### ⚙️ Modular Architecture

The system is broken down into five core modules, orchestrated by a central optimizer. This separation of concerns makes the system maintainable, testable, and highly extensible.

1.  **The Orchestrator (The Brain)**: Manages the main evolutionary loop. It uses a **multi-armed bandit** to intelligently select which mutation strategy to apply in each generation, learning over time what works best for a given problem.

2.  **The Analyzer (The Eyes)**: This is the introspection engine. When given a new script, it reads the source code to understand it.

      * **AST Parser**: Creates a structured representation of the code.
      * **LLM Annotator**: Infers the script's purpose, key hyperparameters, and architecture, translating the code into structured knowledge.

3.  **The Knowledge Graph (The Memory)**: This is the central database that stores the system's understanding of the ML universe.

      * **Ontology**: A predefined schema of ML concepts (`Classifier`, `Layer`, `LossFunction`, etc.).
      * **Graph Database**: Stores concrete instances and relationships, populated both from a pre-compiled base of common libraries and from the **Analyzer's** on-the-fly discoveries.

4.  **The Mutation Engine (The Hands)**: This module performs the actual code transformations. It queries the **Orchestrator's** bandit for which strategy to use and then applies it.

      * **Hyperparameter Mutator**: Tweaks values using Bayesian Optimization.
      * **Structural Mutator**: Swaps components (`RandomForest` -\> `XGBoost`) based on compatibility rules from the **Knowledge Graph**.
      * **LLM Refactor Agent**: Invokes an LLM to perform complex, creative code changes.

5.  **The Executor (The Sandbox)**: This is the firewalled execution environment. It takes a string of mutated code, runs it safely in a separate process with strict timeouts, and reports back the results (metrics, errors, execution time).

-----

### 📂 Final Folder Structure

This folder structure directly mirrors the modular architecture, creating a clean, professional, and scalable project layout.

```
.
├── 📜 README.md
├── 📦 requirements.txt
│
├── automl_lib/
│   ├── __main__.py           # CLI Entry: `python -m automl_lib optimize ...`
│   │
│   ├── orchestration/
│   │   ├── optimizer.py      # The main evolutionary loop.
│   │   └── bandit.py         # The multi-armed bandit for strategy selection.
│   │
│   ├── analysis/
│   │   ├── analyzer.py       # Main interface: script_path -> structured_knowledge.
│   │   ├── ast_parser.py     # Parses code into an AST.
│   │   └── llm_annotator.py  # Uses LLM to infer semantics from code.
│   │
│   ├── knowledge/
│   │   ├── ontology.py       # Defines the schema (nodes, edges) for the KG.
│   │   ├── graph_db.py       # Interface for querying and updating the KG.
│   │   └── precompiled/      # Pre-built KG data for common libs (sklearn, torch).
│   │
│   ├── mutation/
│   │   ├── engine.py         # Main mutation engine, delegates to strategies.
│   │   └── strategies/       # --- The "arms" of the bandit ---
│   │       ├── base_strategy.py
│   │       ├── hyperparameters.py
│   │       ├── structure.py
│   │       └── llm_refactor.py
│   │
│   └── execution/
│       └── sandbox.py        # Manages safe, sandboxed execution of scripts.
│
├── examples/                 # 🚀 User-provided scripts to be optimized.
│   ├── contract.py           # Defines the interface user scripts should follow.
│   ├── simple_classifier.py
│   └── custom_cnn.py
│
└── data/
    └── sample_dataset.csv
```