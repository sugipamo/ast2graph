"""AST to graph conversion library."""

from ast2graph.exceptions import (
    AST2GraphError,
    GraphBuildError,
    ParseError,
    ValidationError,
)
from ast2graph.graph_builder import GraphBuilder
from ast2graph.graph_structure import GraphStructure, SourceInfo
from ast2graph.models import ASTGraphEdge, ASTGraphNode, EdgeType
from ast2graph.parser import ASTParser, ParseResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Main classes
    "GraphBuilder",
    "GraphStructure",
    "SourceInfo",
    "ASTParser",
    "ParseResult",
    # Models
    "ASTGraphNode",
    "ASTGraphEdge", 
    "EdgeType",
    # Exceptions
    "AST2GraphError",
    "ParseError",
    "GraphBuildError",
    "ValidationError",
]
