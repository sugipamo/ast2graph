"""Core data models for ast2graph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class EdgeType(Enum):
    """Types of edges in the AST graph."""
    
    CHILD = "CHILD"                    # Parent-child relationship
    NEXT = "NEXT"                      # Execution order
    DEPENDS_ON = "DEPENDS_ON"          # Dependency relationship
    CALLS = "CALLS"                    # Function call
    DEFINES = "DEFINES"                # Definition relationship
    REFERENCES = "REFERENCES"          # Reference relationship
    IMPORTS = "IMPORTS"                # Import relationship
    USES = "USES"                      # Uses relationship (function/variable)
    INSTANTIATES = "INSTANTIATES"      # Class instantiation relationship


@dataclass
class ASTGraphNode:
    """Represents a node in the AST graph."""
    
    node_id: str                         # Unique node identifier
    node_type: str                       # AST type (ast.AST.__class__.__name__)
    label: str                           # Human-readable label
    ast_node_info: Dict[str, Any]        # Additional AST node information
    source_location: Optional[tuple] = None  # (start_line, start_col, end_line, end_col)
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def __post_init__(self) -> None:
        """Validate the node after initialization."""
        if not self.node_id:
            raise ValueError("node_id cannot be empty")
        
        if not self.node_type:
            raise ValueError("node_type cannot be empty")
        
        if not self.label:
            raise ValueError("label cannot be empty")
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ASTGraphEdge:
    """Represents an edge in the AST graph."""
    
    edge_id: str                         # Unique edge identifier
    source_id: str                       # Source node ID
    target_id: str                       # Target node ID
    edge_type: EdgeType                  # Edge type
    label: str                           # Edge label
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def __post_init__(self) -> None:
        """Validate the edge after initialization."""
        if not self.edge_id:
            raise ValueError("edge_id cannot be empty")
        
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        
        if not isinstance(self.edge_type, EdgeType):
            raise ValueError(f"edge_type must be an EdgeType enum, got {type(self.edge_type)}")
        
        if not self.label:
            raise ValueError("label cannot be empty")
        
        if self.metadata is None:
            self.metadata = {}