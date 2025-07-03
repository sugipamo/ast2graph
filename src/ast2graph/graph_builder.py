"""AST to graph conversion functionality."""

import ast
import uuid
from typing import Any, Dict, Optional

from ast2graph.exceptions import GraphBuildError
from ast2graph.graph_structure import GraphStructure, SourceInfo
from ast2graph.models import ASTGraphEdge, ASTGraphNode, EdgeType


class GraphBuilder:
    """Builds graph structure from Python AST."""

    def __init__(self, ast_tree: ast.AST, source_info: SourceInfo) -> None:
        """Initialize the graph builder.
        
        Args:
            ast_tree: The parsed AST tree
            source_info: Information about the source code
        """
        self.ast_tree = ast_tree
        self.source_info = source_info
        self.graph = GraphStructure()
        self.graph.source_info = source_info  # ソース情報を設定
        self._node_counter = 0
        self._node_mapping: Dict[ast.AST, str] = {}
        # Base UUID for deterministic node ID generation
        self._base_uuid = uuid.UUID('00000000-0000-0000-0000-000000000000')

    def build_graph(self) -> GraphStructure:
        """Build graph from AST.
        
        Returns:
            The constructed graph structure
            
        Raises:
            GraphBuildError: If graph building fails
        """
        try:
            self._visit(self.ast_tree)
            return self.graph
        except Exception as e:
            raise GraphBuildError(
                f"Failed to build graph: {str(e)}",
                node_type=type(self.ast_tree).__name__ if self.ast_tree else None
            ) from e

    def _visit(self, node: ast.AST, parent_id: Optional[str] = None) -> str:
        """Visit an AST node and convert it to a graph node.
        
        Args:
            node: The AST node to visit
            parent_id: The ID of the parent graph node
            
        Returns:
            The ID of the created graph node
        """
        # Create graph node
        graph_node = self._create_node(node)
        node_id = graph_node.node_id
        
        # Add node to graph
        self.graph.add_node(graph_node)
        
        # Map AST node to graph node ID
        self._node_mapping[node] = node_id
        
        # Create edge from parent if exists
        if parent_id is not None:
            edge_type = self._determine_edge_type(parent_id, node_id, node)
            self._create_edge(parent_id, node_id, edge_type)
        
        # Visit child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._visit(item, node_id)
            elif isinstance(value, ast.AST):
                self._visit(value, node_id)
        
        return node_id

    def _create_node(self, ast_node: ast.AST) -> ASTGraphNode:
        """Create a graph node from an AST node.
        
        Args:
            ast_node: The AST node
            
        Returns:
            The created graph node
        """
        node_type = type(ast_node).__name__
        value = self._extract_node_value(ast_node)
        
        # Generate deterministic UUID based on counter
        node_id = self._generate_deterministic_uuid()
        self._node_counter += 1
        
        # Extract position information
        lineno = getattr(ast_node, 'lineno', 0)
        col_offset = getattr(ast_node, 'col_offset', 0)
        end_lineno = getattr(ast_node, 'end_lineno', None)
        end_col_offset = getattr(ast_node, 'end_col_offset', None)
        
        # Create source location tuple
        source_location = None
        if lineno > 0:
            source_location = (lineno, col_offset, end_lineno or lineno, end_col_offset or col_offset)
        
        # Create AST node info dictionary
        ast_node_info = {
            'type': node_type,
            'lineno': lineno,
            'col_offset': col_offset
        }
        
        # Add specific attributes based on node type
        if hasattr(ast_node, 'name'):
            ast_node_info['name'] = ast_node.name
        if hasattr(ast_node, 'id'):
            ast_node_info['id'] = ast_node.id
        if hasattr(ast_node, 'value') and isinstance(ast_node, ast.Constant):
            ast_node_info['value'] = ast_node.value
        
        # Add import-specific attributes
        if isinstance(ast_node, ast.ImportFrom):
            if ast_node.module:
                ast_node_info['module'] = ast_node.module
            ast_node_info['level'] = ast_node.level
        elif isinstance(ast_node, ast.alias):
            ast_node_info['name'] = ast_node.name
            if ast_node.asname:
                ast_node_info['asname'] = ast_node.asname
        
        # Add assignment-specific attributes
        if isinstance(ast_node, ast.Assign):
            # Extract target names
            targets = []
            for target in ast_node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
                elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                    # Handle tuple/list unpacking
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            targets.append(elt.id)
            if targets:
                ast_node_info['targets'] = targets
                # For single target, also set name for compatibility
                if len(targets) == 1:
                    ast_node_info['name'] = targets[0]
        elif isinstance(ast_node, (ast.AugAssign, ast.AnnAssign)):
            # For augmented/annotated assignments
            if hasattr(ast_node, 'target') and isinstance(ast_node.target, ast.Name):
                ast_node_info['target'] = ast_node.target.id
                ast_node_info['name'] = ast_node.target.id
        
        # Add class-specific attributes
        if isinstance(ast_node, ast.ClassDef):
            # Extract base class names
            bases = []
            for base in ast_node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    # Handle cases like ABC.abstractmethod
                    bases.append(f"{self._get_full_attribute_name(base)}")
            if bases:
                ast_node_info['bases'] = bases
        
        return ASTGraphNode(
            node_id=node_id,
            node_type=node_type,
            label=value or node_type,
            ast_node_info=ast_node_info,
            source_location=source_location,
            metadata={}
        )

    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full attribute name including the object.
        
        Args:
            node: The attribute node
            
        Returns:
            The full attribute name (e.g., "module.attr")
        """
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))
    
    def _extract_node_value(self, ast_node: ast.AST) -> Optional[str]:
        """Extract the value from an AST node.
        
        Args:
            ast_node: The AST node
            
        Returns:
            The extracted value or None
        """
        if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return ast_node.name
        elif isinstance(ast_node, ast.Name):
            return ast_node.id
        elif isinstance(ast_node, ast.Constant):
            return str(ast_node.value)
        elif isinstance(ast_node, ast.Attribute):
            return ast_node.attr
        elif isinstance(ast_node, ast.Import):
            # For imports, return the first alias name
            if ast_node.names:
                return ast_node.names[0].name
        elif isinstance(ast_node, ast.ImportFrom):
            return ast_node.module
        elif isinstance(ast_node, ast.alias):
            return ast_node.name
        elif isinstance(ast_node, ast.arg):
            return ast_node.arg
        elif isinstance(ast_node, ast.keyword):
            return ast_node.arg
        elif isinstance(ast_node, (ast.For, ast.While)):
            return "loop"
        elif isinstance(ast_node, ast.If):
            return "conditional"
        elif isinstance(ast_node, (ast.Return, ast.Yield, ast.YieldFrom)):
            return "return"
        elif isinstance(ast_node, ast.Assign):
            # For assignments, try to get the target name
            if ast_node.targets and isinstance(ast_node.targets[0], ast.Name):
                return ast_node.targets[0].id
        elif isinstance(ast_node, (ast.AugAssign, ast.AnnAssign)):
            # For augmented/annotated assignments, get target name
            if hasattr(ast_node, 'target') and isinstance(ast_node.target, ast.Name):
                return ast_node.target.id
        
        return None

    def _create_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        """Create an edge between two nodes.
        
        Args:
            source_id: The source node ID
            target_id: The target node ID
            edge_type: The type of edge
        """
        # Generate deterministic edge ID based on source, target, and edge type
        edge_id = f"edge_{source_id}_{target_id}_{edge_type.value}"
        
        edge = ASTGraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=edge_type.value.lower(),
            metadata={}
        )
        self.graph.add_edge(edge)

    def _determine_edge_type(self, parent_id: str, child_id: str, child_node: ast.AST) -> EdgeType:
        """Determine the edge type between parent and child nodes.
        
        Args:
            parent_id: The parent node ID
            child_id: The child node ID
            child_node: The child AST node
            
        Returns:
            The determined edge type
        """
        # Get parent node type
        parent_node = self.graph.nodes.get(parent_id)
        if not parent_node:
            return EdgeType.CHILD
        
        parent_type = parent_node.node_type
        child_type = type(child_node).__name__
        
        # Determine edge type based on parent-child relationship
        if parent_type in ("Module", "FunctionDef", "AsyncFunctionDef", "ClassDef", "If", "For", "While", "With", "Try"):
            # These are container/control flow nodes
            return EdgeType.CHILD
        elif parent_type in ("Import", "ImportFrom") and child_type == "alias":
            return EdgeType.DEFINES
        elif parent_type == "Assign" and child_type == "Name":
            # Check if it's a target (LHS) or value (RHS)
            parent_ast = next((k for k, v in self._node_mapping.items() if v == parent_id), None)
            if parent_ast and hasattr(parent_ast, 'targets'):
                # If the child is in targets, it's a definition
                child_ast = next((k for k, v in self._node_mapping.items() if v == child_id), None)
                if child_ast in parent_ast.targets:
                    return EdgeType.DEFINES
            return EdgeType.REFERENCES
        elif parent_type in ("AugAssign", "AnnAssign") and child_type == "Name":
            # Check if it's the target
            parent_ast = next((k for k, v in self._node_mapping.items() if v == parent_id), None)
            if parent_ast and hasattr(parent_ast, 'target'):
                child_ast = next((k for k, v in self._node_mapping.items() if v == child_id), None)
                if child_ast == parent_ast.target:
                    return EdgeType.DEFINES
            return EdgeType.REFERENCES
        elif parent_type == "Call":
            # Function calls
            return EdgeType.CALLS
        elif parent_type == "Attribute" and child_type == "Name":
            # Attribute access
            return EdgeType.REFERENCES
        elif child_type == "Name" and parent_type not in ("FunctionDef", "AsyncFunctionDef", "ClassDef", "arg"):
            # General name references
            return EdgeType.REFERENCES
        else:
            # Default to CHILD for structural relationships
            return EdgeType.CHILD
    
    def _generate_deterministic_uuid(self) -> str:
        """Generate a deterministic UUID based on the node counter.
        
        Returns:
            A deterministic UUID string
        """
        # Create a deterministic UUID by modifying the base UUID
        # Use the counter to create unique but deterministic IDs
        int_value = self._base_uuid.int + self._node_counter
        return str(uuid.UUID(int=int_value))