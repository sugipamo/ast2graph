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
    
    def test_edge_type_members(self):
        """Test EdgeType members."""
        expected_members = {"CHILD", "NEXT", "DEPENDS_ON", "CALLS", "DEFINES", "REFERENCES"}
        actual_members = {member.name for member in EdgeType}
        assert actual_members == expected_members


class TestASTGraphNode:
    """Test ASTGraphNode data class."""
    
    def test_create_node_with_all_fields(self):
        """Test creating a node with all fields specified."""
        node_id = str(uuid.uuid4())
        source_id = str(uuid.uuid4())
        
        node = ASTGraphNode(
            id=node_id,
            node_type="FunctionDef",
            value="test_function",
            lineno=10,
            col_offset=4,
            end_lineno=15,
            end_col_offset=0,
            source_id=source_id,
            metadata={"docstring": "Test function"}
        )
        
        assert node.id == node_id
        assert node.node_type == "FunctionDef"
        assert node.value == "test_function"
        assert node.lineno == 10
        assert node.col_offset == 4
        assert node.end_lineno == 15
        assert node.end_col_offset == 0
        assert node.source_id == source_id
        assert node.metadata == {"docstring": "Test function"}
    
    def test_create_node_with_minimal_fields(self):
        """Test creating a node with minimal required fields."""
        source_id = str(uuid.uuid4())
        
        node = ASTGraphNode(
            id="",  # Will be auto-generated
            node_type="Module",
            source_id=source_id
        )
        
        assert node.id  # Should have auto-generated UUID
        assert node.node_type == "Module"
        assert node.value is None
        assert node.lineno == 0
        assert node.col_offset == 0
        assert node.end_lineno is None
        assert node.end_col_offset is None
        assert node.source_id == source_id
        assert node.metadata == {}
    
    def test_auto_generate_uuid(self):
        """Test that UUID is auto-generated when not provided."""
        node = ASTGraphNode(
            id="",
            node_type="Module",
            source_id=str(uuid.uuid4())
        )
        
        # Verify it's a valid UUID
        uuid.UUID(node.id)  # Should not raise
    
    def test_invalid_uuid_format(self):
        """Test that invalid UUID format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID format"):
            ASTGraphNode(
                id="not-a-uuid",
                node_type="Module",
                source_id=str(uuid.uuid4())
            )
    
    def test_empty_node_type(self):
        """Test that empty node_type raises ValueError."""
        with pytest.raises(ValueError, match="node_type cannot be empty"):
            ASTGraphNode(
                id="",
                node_type="",
                source_id=str(uuid.uuid4())
            )
    
    def test_empty_source_id(self):
        """Test that empty source_id raises ValueError."""
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            ASTGraphNode(
                id="",
                node_type="Module",
                source_id=""
            )
    
    def test_negative_line_numbers(self):
        """Test that negative line numbers raise ValueError."""
        source_id = str(uuid.uuid4())
        
        with pytest.raises(ValueError, match="lineno cannot be negative"):
            ASTGraphNode(
                id="",
                node_type="Module",
                lineno=-1,
                source_id=source_id
            )
        
        with pytest.raises(ValueError, match="col_offset cannot be negative"):
            ASTGraphNode(
                id="",
                node_type="Module",
                col_offset=-1,
                source_id=source_id
            )
        
        with pytest.raises(ValueError, match="end_lineno cannot be negative"):
            ASTGraphNode(
                id="",
                node_type="Module",
                end_lineno=-1,
                source_id=source_id
            )
        
        with pytest.raises(ValueError, match="end_col_offset cannot be negative"):
            ASTGraphNode(
                id="",
                node_type="Module",
                end_col_offset=-1,
                source_id=source_id
            )


class TestASTGraphEdge:
    """Test ASTGraphEdge data class."""
    
    def test_create_edge_with_all_fields(self):
        """Test creating an edge with all fields specified."""
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        
        edge = ASTGraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.CHILD,
            properties={"index": 0},
            order=1
        )
        
        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.edge_type == EdgeType.CHILD
        assert edge.properties == {"index": 0}
        assert edge.order == 1
    
    def test_create_edge_with_minimal_fields(self):
        """Test creating an edge with minimal required fields."""
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        
        edge = ASTGraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.DEPENDS_ON
        )
        
        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.edge_type == EdgeType.DEPENDS_ON
        assert edge.properties == {}
        assert edge.order is None
    
    def test_empty_source_id(self):
        """Test that empty source_id raises ValueError."""
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            ASTGraphEdge(
                source_id="",
                target_id=str(uuid.uuid4()),
                edge_type=EdgeType.CHILD
            )
    
    def test_empty_target_id(self):
        """Test that empty target_id raises ValueError."""
        with pytest.raises(ValueError, match="target_id cannot be empty"):
            ASTGraphEdge(
                source_id=str(uuid.uuid4()),
                target_id="",
                edge_type=EdgeType.CHILD
            )
    
    def test_invalid_source_uuid(self):
        """Test that invalid source UUID format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID format for source_id"):
            ASTGraphEdge(
                source_id="not-a-uuid",
                target_id=str(uuid.uuid4()),
                edge_type=EdgeType.CHILD
            )
    
    def test_invalid_target_uuid(self):
        """Test that invalid target UUID format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID format for target_id"):
            ASTGraphEdge(
                source_id=str(uuid.uuid4()),
                target_id="not-a-uuid",
                edge_type=EdgeType.CHILD
            )
    
    def test_invalid_edge_type(self):
        """Test that invalid edge_type raises ValueError."""
        with pytest.raises(ValueError, match="edge_type must be an EdgeType enum"):
            ASTGraphEdge(
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                edge_type="INVALID"  # type: ignore
            )
    
    def test_negative_order_for_next_edge(self):
        """Test that negative order for NEXT edges raises ValueError."""
        with pytest.raises(ValueError, match="order cannot be negative for NEXT edges"):
            ASTGraphEdge(
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                edge_type=EdgeType.NEXT,
                order=-1
            )
    
    def test_negative_order_allowed_for_non_next_edge(self):
        """Test that negative order is allowed for non-NEXT edges."""
        # Should not raise
        edge = ASTGraphEdge(
            source_id=str(uuid.uuid4()),
            target_id=str(uuid.uuid4()),
            edge_type=EdgeType.CHILD,
            order=-1
        )
        assert edge.order == -1