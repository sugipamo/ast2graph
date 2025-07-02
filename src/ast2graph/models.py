"""Core data models for ast2graph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class EdgeType(Enum):
    """Types of edges in the AST graph."""
    
    CHILD = "CHILD"                    # Parent-child relationship
    NEXT = "NEXT"                      # Execution order
    DEPENDS_ON = "DEPENDS_ON"          # Dependency relationship
    CALLS = "CALLS"                    # Function call
    DEFINES = "DEFINES"                # Definition relationship
    REFERENCES = "REFERENCES"          # Reference relationship


@dataclass
class ASTGraphNode:
    """Represents a node in the AST graph."""
    
    id: str                              # UUID4 format
    node_type: str                       # AST type (ast.AST.__class__.__name__)
    value: Optional[str] = None          # Node value (variable name, function name, etc.)
    lineno: int = 0                      # Line number
    col_offset: int = 0                  # Column offset
    end_lineno: Optional[int] = None     # End line number
    end_col_offset: Optional[int] = None # End column offset
    source_id: str = ""                  # Source file identifier
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate the node after initialization."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Validate UUID format
        try:
            uuid.UUID(self.id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for node id: {self.id}")
        
        if not self.node_type:
            raise ValueError("node_type cannot be empty")
        
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        
        # Ensure line numbers are non-negative
        if self.lineno < 0:
            raise ValueError(f"lineno cannot be negative: {self.lineno}")
        
        if self.col_offset < 0:
            raise ValueError(f"col_offset cannot be negative: {self.col_offset}")
        
        if self.end_lineno is not None and self.end_lineno < 0:
            raise ValueError(f"end_lineno cannot be negative: {self.end_lineno}")
        
        if self.end_col_offset is not None and self.end_col_offset < 0:
            raise ValueError(f"end_col_offset cannot be negative: {self.end_col_offset}")


@dataclass
class ASTGraphEdge:
    """Represents an edge in the AST graph."""
    
    source_id: str                       # Source node ID
    target_id: str                       # Target node ID
    edge_type: EdgeType                  # Edge type
    properties: Dict[str, Any] = field(default_factory=dict)
    order: Optional[int] = None          # Execution order (for NEXT edges)
    
    def __post_init__(self) -> None:
        """Validate the edge after initialization."""
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        
        # Validate UUID format for source and target
        try:
            uuid.UUID(self.source_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for source_id: {self.source_id}")
        
        try:
            uuid.UUID(self.target_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for target_id: {self.target_id}")
        
        if not isinstance(self.edge_type, EdgeType):
            raise ValueError(f"edge_type must be an EdgeType enum, got {type(self.edge_type)}")
        
        # Validate order for NEXT edges
        if self.edge_type == EdgeType.NEXT and self.order is not None and self.order < 0:
            raise ValueError(f"order cannot be negative for NEXT edges: {self.order}")