"""Graph structure management for ast2graph."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import uuid

from .models import ASTGraphNode, ASTGraphEdge, EdgeType


@dataclass
class SourceInfo:
    """Information about the source file."""
    
    source_id: str                      # Source file identifier (UUID)
    file_path: str                      # File path
    file_hash: str                      # File hash (SHA256)
    parsed_at: datetime                 # Parse timestamp
    encoding: str = "utf-8"             # Encoding
    line_count: int = 0                 # Line count
    size_bytes: int = 0                 # File size in bytes
    
    def __post_init__(self) -> None:
        """Validate source info after initialization."""
        if not self.source_id:
            self.source_id = str(uuid.uuid4())
        
        # Validate UUID format
        try:
            uuid.UUID(self.source_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for source_id: {self.source_id}")
        
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        
        if not self.file_hash:
            raise ValueError("file_hash cannot be empty")
        
        # Validate SHA256 hash format (64 hex characters)
        if len(self.file_hash) != 64 or not all(c in '0123456789abcdef' for c in self.file_hash.lower()):
            raise ValueError(f"Invalid SHA256 hash format: {self.file_hash}")
        
        if self.line_count < 0:
            raise ValueError(f"line_count cannot be negative: {self.line_count}")
        
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes cannot be negative: {self.size_bytes}")
    
    @classmethod
    def from_content(cls, file_path: str, content: bytes, encoding: str = "utf-8") -> "SourceInfo":
        """Create SourceInfo from file content."""
        file_hash = hashlib.sha256(content).hexdigest()
        line_count = content.decode(encoding, errors='ignore').count('\n') + 1
        
        return cls(
            source_id=str(uuid.uuid4()),
            file_path=file_path,
            file_hash=file_hash,
            parsed_at=datetime.now(),
            encoding=encoding,
            line_count=line_count,
            size_bytes=len(content)
        )


@dataclass
class GraphStructure:
    """Main graph structure containing nodes and edges."""
    
    nodes: Dict[str, ASTGraphNode] = field(default_factory=dict)
    edges: List[ASTGraphEdge] = field(default_factory=list)
    source_info: Optional[SourceInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize internal structures."""
        # Create edge indices for efficient lookup
        self._edges_from: Dict[str, List[ASTGraphEdge]] = {}
        self._edges_to: Dict[str, List[ASTGraphEdge]] = {}
        
        # Rebuild indices if edges exist
        for edge in self.edges:
            self._add_edge_to_indices(edge)
    
    def _add_edge_to_indices(self, edge: ASTGraphEdge) -> None:
        """Add edge to internal indices."""
        if edge.source_id not in self._edges_from:
            self._edges_from[edge.source_id] = []
        self._edges_from[edge.source_id].append(edge)
        
        if edge.target_id not in self._edges_to:
            self._edges_to[edge.target_id] = []
        self._edges_to[edge.target_id].append(edge)
    
    def add_node(self, node: ASTGraphNode) -> None:
        """Add a node to the graph."""
        if not isinstance(node, ASTGraphNode):
            raise ValueError(f"Expected ASTGraphNode, got {type(node)}")
        
        if node.node_id in self.nodes:
            raise ValueError(f"Node with id {node.node_id} already exists")
        
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: ASTGraphEdge) -> None:
        """Add an edge to the graph."""
        if not isinstance(edge, ASTGraphEdge):
            raise ValueError(f"Expected ASTGraphEdge, got {type(edge)}")
        
        # Validate that both nodes exist
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist")
        
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} does not exist")
        
        self.edges.append(edge)
        self._add_edge_to_indices(edge)
    
    def get_node(self, node_id: str) -> Optional[ASTGraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[ASTGraphEdge]:
        """Get all edges originating from a node."""
        return self._edges_from.get(node_id, [])
    
    def get_edges_to(self, node_id: str) -> List[ASTGraphEdge]:
        """Get all edges pointing to a node."""
        return self._edges_to.get(node_id, [])
    
    def get_edges_of_type(self, edge_type: EdgeType) -> List[ASTGraphEdge]:
        """Get all edges of a specific type."""
        return [edge for edge in self.edges if edge.edge_type == edge_type]
    
    def get_children(self, node_id: str) -> List[ASTGraphNode]:
        """Get all child nodes of a node."""
        child_edges = [e for e in self.get_edges_from(node_id) if e.edge_type == EdgeType.CHILD]
        return [self.nodes[e.target_id] for e in child_edges if e.target_id in self.nodes]
    
    def get_parent(self, node_id: str) -> Optional[ASTGraphNode]:
        """Get the parent node of a node."""
        parent_edges = [e for e in self.get_edges_to(node_id) if e.edge_type == EdgeType.CHILD]
        if parent_edges:
            return self.nodes.get(parent_edges[0].source_id)
        return None
    
    def validate(self) -> List[str]:
        """Validate the graph structure and return list of errors."""
        errors = []
        
        # Check if source_info is set
        if not self.source_info:
            errors.append("source_info is not set")
        
        # Check for orphaned edges
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                errors.append(f"Edge references non-existent source node: {edge.source_id}")
            if edge.target_id not in self.nodes:
                errors.append(f"Edge references non-existent target node: {edge.target_id}")
        
        # Check for duplicate edges
        edge_set = set()
        for edge in self.edges:
            edge_key = (edge.source_id, edge.target_id, edge.edge_type)
            if edge_key in edge_set:
                errors.append(f"Duplicate edge found: {edge_key}")
            edge_set.add(edge_key)
        
        # Check for cycles in CHILD relationships
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for edge in self.get_edges_from(node_id):
                if edge.edge_type == EdgeType.CHILD:
                    if edge.target_id not in visited:
                        if has_cycle(edge.target_id):
                            return True
                    elif edge.target_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append("Cycle detected in CHILD relationships")
                    break
        
        # Check that all nodes have the same source_id
        if self.source_info:
            for node in self.nodes.values():
                if node.source_id != self.source_info.source_id:
                    errors.append(f"Node {node.id} has different source_id than graph: {node.source_id}")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        edge_type_counts = {}
        for edge in self.edges:
            edge_type_counts[edge.edge_type.value] = edge_type_counts.get(edge.edge_type.value, 0) + 1
        
        node_type_counts = {}
        for node in self.nodes.values():
            node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
        
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": node_type_counts,
            "edge_types": edge_type_counts,
            "has_source_info": self.source_info is not None
        }