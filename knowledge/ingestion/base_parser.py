# automl_lib/knowledge/ingestion/base_parser.py
"""
Defines the abstract base class for all library ingestion parsers.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from knowledge.ontology import Node, Relationship

class BaseParser(ABC):
    """
    An interface for a parser that can inspect a Python library and extract
    a knowledge graph representation of its components.
    """
    @abstractmethod
    def parse(self) -> Iterator[Tuple[Node, list[Relationship]]]:
        """
        Parses the target library and yields its components and relationships.

        Yields:
            A tuple containing a Node (like a ComponentNode or DataTypeNode)
            and a list of Relationship objects associated with that node.
        """
        pass