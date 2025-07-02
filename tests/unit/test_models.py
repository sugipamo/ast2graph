"""Unit tests for models module."""

import pytest
from datetime import datetime
import uuid

from ast2graph.models import EdgeType, ASTGraphNode, ASTGraphEdge


class TestEdgeType:
    """Test EdgeType enumeration."""
    
    def test_edge_type_values(self):
        """Test that EdgeType has expected values."""
        assert EdgeType.CHILD.value == "CHILD"
        assert EdgeType.NEXT.value == "NEXT"
        assert EdgeType.DEPENDS_ON.value == "DEPENDS_ON"
        assert EdgeType.CALLS.value == "CALLS"
        assert EdgeType.DEFINES.value == "DEFINES"
        assert EdgeType.REFERENCES.value == "REFERENCES"
        assert EdgeType.IMPORTS.value == "IMPORTS"
        assert EdgeType.USES.value == "USES"
        assert EdgeType.INSTANTIATES.value == "INSTANTIATES"
    
    def test_edge_type_members(self):
        """Test EdgeType members."""
        expected_members = {"CHILD", "NEXT", "DEPENDS_ON", "CALLS", "DEFINES", "REFERENCES", "IMPORTS", "USES", "INSTANTIATES"}
        actual_members = {member.name for member in EdgeType}
        assert actual_members == expected_members


class TestASTGraphNode:
    """Test ASTGraphNode data class."""
    
    def test_create_node_with_all_fields(self):
        """Test creating a node with all fields specified."""
        node_id = str(uuid.uuid4())
        source_id = str(uuid.uuid4())
        
        node = ASTGraphNode(
            node_id=node_id,
            node_type="FunctionDef",
            label="test_function",
            ast_node_info={"source_id": source_id},
            source_location=(10, 4, 15, 0),
            metadata={"docstring": "Test function"}
        )
        
        assert node.node_id == node_id
        assert node.node_type == "FunctionDef"
        assert node.label == "test_function"
        assert node.source_location == (10, 4, 15, 0)
        assert node.ast_node_info == {"source_id": source_id}
        assert node.metadata == {"docstring": "Test function"}
    
    def test_create_node_with_minimal_fields(self):
        """Test creating a node with minimal required fields."""
        node_id = str(uuid.uuid4())
        
        node = ASTGraphNode(
            node_id=node_id,
            node_type="Module",
            label="Module",
            ast_node_info={}
        )
        
        assert node.node_id == node_id
        assert node.node_type == "Module"
        assert node.label == "Module"
        assert node.source_location is None
        assert node.ast_node_info == {}
        assert node.metadata == {}
    
    def test_empty_node_id(self):
        """Test that empty node_id raises ValueError."""
        with pytest.raises(ValueError, match="node_id cannot be empty"):
            ASTGraphNode(
                node_id="",
                node_type="Module",
                label="Module",
                ast_node_info={}
            )
    
    def test_empty_node_type(self):
        """Test that empty node_type raises ValueError."""
        with pytest.raises(ValueError, match="node_type cannot be empty"):
            ASTGraphNode(
                node_id=str(uuid.uuid4()),
                node_type="",
                label="Module",
                ast_node_info={}
            )
    
    def test_empty_label(self):
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="label cannot be empty"):
            ASTGraphNode(
                node_id=str(uuid.uuid4()),
                node_type="Module",
                label="",
                ast_node_info={}
            )
    
    def test_node_with_source_location(self):
        """Test node with source location tuple."""
        node_id = str(uuid.uuid4())
        
        node = ASTGraphNode(
            node_id=node_id,
            node_type="FunctionDef",
            label="test_func",
            ast_node_info={},
            source_location=(1, 0, 5, 10)
        )
        
        assert node.source_location == (1, 0, 5, 10)
    
    def test_node_with_none_metadata(self):
        """Test that None metadata is converted to empty dict."""
        node = ASTGraphNode(
            node_id=str(uuid.uuid4()),
            node_type="Module",
            label="Module",
            ast_node_info={},
            metadata=None
        )
        
        assert node.metadata == {}


class TestASTGraphEdge:
    """Test ASTGraphEdge data class."""
    
    def test_create_edge_with_all_fields(self):
        """Test creating an edge with all fields specified."""
        edge_id = str(uuid.uuid4())
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        
        edge = ASTGraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.CHILD,
            label="child_0",
            metadata={"index": 0}
        )
        
        assert edge.edge_id == edge_id
        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.edge_type == EdgeType.CHILD
        assert edge.label == "child_0"
        assert edge.metadata == {"index": 0}
    
    def test_create_edge_with_minimal_fields(self):
        """Test creating an edge with minimal required fields."""
        edge_id = str(uuid.uuid4())
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        
        edge = ASTGraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.DEPENDS_ON,
            label="depends_on"
        )
        
        assert edge.edge_id == edge_id
        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.edge_type == EdgeType.DEPENDS_ON
        assert edge.label == "depends_on"
        assert edge.metadata == {}
    
    def test_empty_edge_id(self):
        """Test that empty edge_id raises ValueError."""
        with pytest.raises(ValueError, match="edge_id cannot be empty"):
            ASTGraphEdge(
                edge_id="",
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                edge_type=EdgeType.CHILD,
                label="child"
            )
    
    def test_empty_source_id(self):
        """Test that empty source_id raises ValueError."""
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            ASTGraphEdge(
                edge_id=str(uuid.uuid4()),
                source_id="",
                target_id=str(uuid.uuid4()),
                edge_type=EdgeType.CHILD,
                label="child"
            )
    
    def test_empty_target_id(self):
        """Test that empty target_id raises ValueError."""
        with pytest.raises(ValueError, match="target_id cannot be empty"):
            ASTGraphEdge(
                edge_id=str(uuid.uuid4()),
                source_id=str(uuid.uuid4()),
                target_id="",
                edge_type=EdgeType.CHILD,
                label="child"
            )
    
    def test_invalid_edge_type(self):
        """Test that invalid edge_type raises ValueError."""
        with pytest.raises(ValueError, match="edge_type must be an EdgeType enum"):
            ASTGraphEdge(
                edge_id=str(uuid.uuid4()),
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                edge_type="INVALID",  # type: ignore
                label="invalid"
            )
    
    def test_empty_label(self):
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="label cannot be empty"):
            ASTGraphEdge(
                edge_id=str(uuid.uuid4()),
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                edge_type=EdgeType.CHILD,
                label=""
            )
    
    def test_edge_with_none_metadata(self):
        """Test that None metadata is converted to empty dict."""
        edge = ASTGraphEdge(
            edge_id=str(uuid.uuid4()),
            source_id=str(uuid.uuid4()),
            target_id=str(uuid.uuid4()),
            edge_type=EdgeType.NEXT,
            label="next",
            metadata=None
        )
        
        assert edge.metadata == {}