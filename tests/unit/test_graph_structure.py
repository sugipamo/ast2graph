"""Unit tests for graph_structure module."""

import hashlib
import uuid
from datetime import datetime

import pytest

from ast2graph.graph_structure import GraphStructure, SourceInfo
from ast2graph.models import ASTGraphEdge, ASTGraphNode, EdgeType


class TestSourceInfo:
    """Test SourceInfo data class."""

    def test_create_source_info_with_all_fields(self):
        """Test creating SourceInfo with all fields specified."""
        source_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(b"test content").hexdigest()
        parsed_at = datetime.now()

        info = SourceInfo(
            source_id=source_id,
            file_path="/path/to/file.py",
            file_hash=file_hash,
            parsed_at=parsed_at,
            encoding="utf-8",
            line_count=100,
            size_bytes=1024
        )

        assert info.source_id == source_id
        assert info.file_path == "/path/to/file.py"
        assert info.file_hash == file_hash
        assert info.parsed_at == parsed_at
        assert info.encoding == "utf-8"
        assert info.line_count == 100
        assert info.size_bytes == 1024

    def test_auto_generate_uuid(self):
        """Test that UUID is auto-generated when not provided."""
        file_hash = hashlib.sha256(b"test").hexdigest()

        info = SourceInfo(
            source_id="",
            file_path="/test.py",
            file_hash=file_hash,
            parsed_at=datetime.now()
        )

        # Verify it's a valid UUID
        uuid.UUID(info.source_id)  # Should not raise

    def test_invalid_uuid_format(self):
        """Test that invalid UUID format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID format"):
            SourceInfo(
                source_id="not-a-uuid",
                file_path="/test.py",
                file_hash=hashlib.sha256(b"test").hexdigest(),
                parsed_at=datetime.now()
            )

    def test_empty_file_path(self):
        """Test that empty file_path raises ValueError."""
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            SourceInfo(
                source_id="",
                file_path="",
                file_hash=hashlib.sha256(b"test").hexdigest(),
                parsed_at=datetime.now()
            )

    def test_empty_file_hash(self):
        """Test that empty file_hash raises ValueError."""
        with pytest.raises(ValueError, match="file_hash cannot be empty"):
            SourceInfo(
                source_id="",
                file_path="/test.py",
                file_hash="",
                parsed_at=datetime.now()
            )

    def test_invalid_sha256_hash(self):
        """Test that invalid SHA256 hash format raises ValueError."""
        # Too short
        with pytest.raises(ValueError, match="Invalid SHA256 hash format"):
            SourceInfo(
                source_id="",
                file_path="/test.py",
                file_hash="abc123",
                parsed_at=datetime.now()
            )

        # Invalid characters
        with pytest.raises(ValueError, match="Invalid SHA256 hash format"):
            SourceInfo(
                source_id="",
                file_path="/test.py",
                file_hash="g" * 64,  # 'g' is not a hex character
                parsed_at=datetime.now()
            )

    def test_negative_line_count(self):
        """Test that negative line_count raises ValueError."""
        with pytest.raises(ValueError, match="line_count cannot be negative"):
            SourceInfo(
                source_id="",
                file_path="/test.py",
                file_hash=hashlib.sha256(b"test").hexdigest(),
                parsed_at=datetime.now(),
                line_count=-1
            )

    def test_negative_size_bytes(self):
        """Test that negative size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="size_bytes cannot be negative"):
            SourceInfo(
                source_id="",
                file_path="/test.py",
                file_hash=hashlib.sha256(b"test").hexdigest(),
                parsed_at=datetime.now(),
                size_bytes=-1
            )

    def test_from_content(self):
        """Test creating SourceInfo from file content."""
        content = b"def hello():\n    print('Hello, world!')\n"
        file_path = "/test.py"

        info = SourceInfo.from_content(file_path, content)

        assert info.file_path == file_path
        assert info.file_hash == hashlib.sha256(content).hexdigest()
        assert info.encoding == "utf-8"
        assert info.line_count == 3  # Two lines + final newline
        assert info.size_bytes == len(content)
        assert isinstance(info.parsed_at, datetime)
        # Verify UUID is generated
        uuid.UUID(info.source_id)  # Should not raise


class TestGraphStructure:
    """Test GraphStructure class."""

    def create_test_node(self, node_id: str = None, node_type: str = "Module", source_id: str = None) -> ASTGraphNode:
        """Helper to create a test node."""
        generated_id = node_id or str(uuid.uuid4())
        return ASTGraphNode(
            node_id=generated_id,
            node_type=node_type,
            label=f"{node_type}_node",
            ast_node_info={
                "source_id": source_id or str(uuid.uuid4()),
                "example_field": "example_value"
            }
        )

    def create_test_edge(self, source_id: str, target_id: str, edge_type: EdgeType, edge_id: str = None) -> ASTGraphEdge:
        """Helper to create a test edge."""
        generated_id = edge_id or str(uuid.uuid4())
        return ASTGraphEdge(
            edge_id=generated_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=f"{edge_type.value}_edge"
        )

    def test_create_empty_graph(self):
        """Test creating an empty graph structure."""
        graph = GraphStructure()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.source_info is None
        assert graph.metadata == {}

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = GraphStructure()
        node1 = self.create_test_node()
        node2 = self.create_test_node()

        graph.add_node(node1)
        graph.add_node(node2)

        assert len(graph.nodes) == 2
        assert graph.nodes[node1.node_id] == node1
        assert graph.nodes[node2.node_id] == node2

    def test_add_duplicate_node(self):
        """Test that adding a duplicate node raises ValueError."""
        graph = GraphStructure()
        node = self.create_test_node()

        graph.add_node(node)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node)

    def test_add_invalid_node_type(self):
        """Test that adding non-ASTGraphNode raises ValueError."""
        graph = GraphStructure()

        with pytest.raises(ValueError, match="Expected ASTGraphNode"):
            graph.add_node("not a node")  # type: ignore

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = GraphStructure()
        node1 = self.create_test_node()
        node2 = self.create_test_node()

        graph.add_node(node1)
        graph.add_node(node2)

        edge = self.create_test_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.CHILD
        )

        graph.add_edge(edge)

        assert len(graph.edges) == 1
        assert graph.edges[0] == edge

    def test_add_edge_missing_source_node(self):
        """Test that adding edge with missing source node raises ValueError."""
        graph = GraphStructure()
        node = self.create_test_node()
        graph.add_node(node)

        edge = self.create_test_edge(
            source_id=str(uuid.uuid4()),  # Non-existent
            target_id=node.node_id,
            edge_type=EdgeType.CHILD
        )

        with pytest.raises(ValueError, match="Source node .* does not exist"):
            graph.add_edge(edge)

    def test_add_edge_missing_target_node(self):
        """Test that adding edge with missing target node raises ValueError."""
        graph = GraphStructure()
        node = self.create_test_node()
        graph.add_node(node)

        edge = self.create_test_edge(
            source_id=node.node_id,
            target_id=str(uuid.uuid4()),  # Non-existent
            edge_type=EdgeType.CHILD
        )

        with pytest.raises(ValueError, match="Target node .* does not exist"):
            graph.add_edge(edge)

    def test_get_node(self):
        """Test getting a node by ID."""
        graph = GraphStructure()
        node = self.create_test_node()
        graph.add_node(node)

        retrieved = graph.get_node(node.node_id)
        assert retrieved == node

        # Non-existent node returns None
        assert graph.get_node(str(uuid.uuid4())) is None

    def test_get_edges_from(self):
        """Test getting edges from a node."""
        graph = GraphStructure()
        node1 = self.create_test_node()
        node2 = self.create_test_node()
        node3 = self.create_test_node()

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        edge1 = self.create_test_edge(node1.node_id, node2.node_id, EdgeType.CHILD)
        edge2 = self.create_test_edge(node1.node_id, node3.node_id, EdgeType.NEXT)
        edge3 = self.create_test_edge(node2.node_id, node3.node_id, EdgeType.DEPENDS_ON)

        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)

        edges_from_node1 = graph.get_edges_from(node1.node_id)
        assert len(edges_from_node1) == 2
        assert edge1 in edges_from_node1
        assert edge2 in edges_from_node1

        edges_from_node2 = graph.get_edges_from(node2.node_id)
        assert len(edges_from_node2) == 1
        assert edge3 in edges_from_node2

        # Non-existent node
        assert graph.get_edges_from(str(uuid.uuid4())) == []

    def test_get_edges_to(self):
        """Test getting edges to a node."""
        graph = GraphStructure()
        node1 = self.create_test_node()
        node2 = self.create_test_node()
        node3 = self.create_test_node()

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        edge1 = self.create_test_edge(node1.node_id, node3.node_id, EdgeType.CHILD)
        edge2 = self.create_test_edge(node2.node_id, node3.node_id, EdgeType.NEXT)

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        edges_to_node3 = graph.get_edges_to(node3.node_id)
        assert len(edges_to_node3) == 2
        assert edge1 in edges_to_node3
        assert edge2 in edges_to_node3

        # Node with no incoming edges
        assert graph.get_edges_to(node1.node_id) == []

    def test_get_edges_of_type(self):
        """Test getting edges of a specific type."""
        graph = GraphStructure()
        node1 = self.create_test_node()
        node2 = self.create_test_node()
        node3 = self.create_test_node()

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        edge1 = self.create_test_edge(node1.node_id, node2.node_id, EdgeType.CHILD)
        edge2 = self.create_test_edge(node2.node_id, node3.node_id, EdgeType.CHILD)
        edge3 = self.create_test_edge(node1.node_id, node3.node_id, EdgeType.DEPENDS_ON)

        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)

        child_edges = graph.get_edges_of_type(EdgeType.CHILD)
        assert len(child_edges) == 2
        assert edge1 in child_edges
        assert edge2 in child_edges

        depends_edges = graph.get_edges_of_type(EdgeType.DEPENDS_ON)
        assert len(depends_edges) == 1
        assert edge3 in depends_edges

    def test_get_children(self):
        """Test getting child nodes."""
        graph = GraphStructure()
        parent = self.create_test_node(node_type="ClassDef")
        child1 = self.create_test_node(node_type="FunctionDef")
        child2 = self.create_test_node(node_type="Assign")
        non_child = self.create_test_node(node_type="Import")

        graph.add_node(parent)
        graph.add_node(child1)
        graph.add_node(child2)
        graph.add_node(non_child)

        graph.add_edge(self.create_test_edge(parent.node_id, child1.node_id, EdgeType.CHILD))
        graph.add_edge(self.create_test_edge(parent.node_id, child2.node_id, EdgeType.CHILD))
        graph.add_edge(self.create_test_edge(parent.node_id, non_child.node_id, EdgeType.DEPENDS_ON))

        children = graph.get_children(parent.node_id)
        assert len(children) == 2
        assert child1 in children
        assert child2 in children
        assert non_child not in children

    def test_get_parent(self):
        """Test getting parent node."""
        graph = GraphStructure()
        parent = self.create_test_node(node_type="Module")
        child = self.create_test_node(node_type="FunctionDef")

        graph.add_node(parent)
        graph.add_node(child)
        graph.add_edge(self.create_test_edge(parent.node_id, child.node_id, EdgeType.CHILD))

        assert graph.get_parent(child.node_id) == parent
        assert graph.get_parent(parent.node_id) is None

    def test_validate_empty_graph(self):
        """Test validating an empty graph."""
        graph = GraphStructure()
        errors = graph.validate()

        assert "source_info is not set" in errors

    def test_validate_with_source_info(self):
        """Test validating a graph with source info."""
        graph = GraphStructure()
        graph.source_info = SourceInfo(
            source_id=str(uuid.uuid4()),
            file_path="/test.py",
            file_hash=hashlib.sha256(b"test").hexdigest(),
            parsed_at=datetime.now()
        )

        errors = graph.validate()
        assert len(errors) == 0

    def test_validate_orphaned_edges(self):
        """Test validation detects orphaned edges."""
        graph = GraphStructure()
        graph.source_info = SourceInfo(
            source_id=str(uuid.uuid4()),
            file_path="/test.py",
            file_hash=hashlib.sha256(b"test").hexdigest(),
            parsed_at=datetime.now()
        )

        # Add edge without nodes
        graph.edges.append(self.create_test_edge(
            source_id=str(uuid.uuid4()),
            target_id=str(uuid.uuid4()),
            edge_type=EdgeType.CHILD
        ))

        errors = graph.validate()
        assert any("non-existent source node" in e for e in errors)
        assert any("non-existent target node" in e for e in errors)

    def test_validate_duplicate_edges(self):
        """Test validation detects duplicate edges."""
        graph = GraphStructure()
        source_id = str(uuid.uuid4())
        graph.source_info = SourceInfo(
            source_id=source_id,
            file_path="/test.py",
            file_hash=hashlib.sha256(b"test").hexdigest(),
            parsed_at=datetime.now()
        )

        node1 = self.create_test_node(source_id=source_id)
        node2 = self.create_test_node(source_id=source_id)

        graph.add_node(node1)
        graph.add_node(node2)

        # Add the same edge twice
        edge = self.create_test_edge(node1.node_id, node2.node_id, EdgeType.CHILD)
        graph.add_edge(edge)
        graph.edges.append(edge)  # Bypass add_edge to create duplicate

        errors = graph.validate()
        assert any("Duplicate edge found" in e for e in errors)

    def test_validate_cycle_detection(self):
        """Test validation detects cycles in CHILD relationships."""
        graph = GraphStructure()
        source_id = str(uuid.uuid4())
        graph.source_info = SourceInfo(
            source_id=source_id,
            file_path="/test.py",
            file_hash=hashlib.sha256(b"test").hexdigest(),
            parsed_at=datetime.now()
        )

        node1 = self.create_test_node(source_id=source_id)
        node2 = self.create_test_node(source_id=source_id)
        node3 = self.create_test_node(source_id=source_id)

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        graph.add_edge(self.create_test_edge(node1.node_id, node2.node_id, EdgeType.CHILD))
        graph.add_edge(self.create_test_edge(node2.node_id, node3.node_id, EdgeType.CHILD))
        graph.add_edge(self.create_test_edge(node3.node_id, node1.node_id, EdgeType.CHILD))  # Creates cycle

        errors = graph.validate()
        assert any("Cycle detected" in e for e in errors)

    def test_validate_mismatched_source_id(self):
        """Test validation detects nodes with mismatched source_id."""
        graph = GraphStructure()
        source_id = str(uuid.uuid4())
        graph.source_info = SourceInfo(
            source_id=source_id,
            file_path="/test.py",
            file_hash=hashlib.sha256(b"test").hexdigest(),
            parsed_at=datetime.now()
        )

        # Node with different source_id
        node = self.create_test_node(source_id=str(uuid.uuid4()))
        graph.add_node(node)

        errors = graph.validate()
        assert any("different source_id than graph" in e for e in errors)

    def test_get_statistics(self):
        """Test getting graph statistics."""
        graph = GraphStructure()
        source_id = str(uuid.uuid4())
        graph.source_info = SourceInfo(
            source_id=source_id,
            file_path="/test.py",
            file_hash=hashlib.sha256(b"test").hexdigest(),
            parsed_at=datetime.now()
        )

        # Add nodes of different types
        module = self.create_test_node(node_type="Module", source_id=source_id)
        func1 = self.create_test_node(node_type="FunctionDef", source_id=source_id)
        func2 = self.create_test_node(node_type="FunctionDef", source_id=source_id)
        assign = self.create_test_node(node_type="Assign", source_id=source_id)

        graph.add_node(module)
        graph.add_node(func1)
        graph.add_node(func2)
        graph.add_node(assign)

        # Add edges of different types
        graph.add_edge(self.create_test_edge(module.node_id, func1.node_id, EdgeType.CHILD))
        graph.add_edge(self.create_test_edge(module.node_id, func2.node_id, EdgeType.CHILD))
        graph.add_edge(self.create_test_edge(func1.node_id, func2.node_id, EdgeType.CALLS))
        graph.add_edge(self.create_test_edge(module.node_id, assign.node_id, EdgeType.CHILD))

        stats = graph.get_statistics()

        assert stats["node_count"] == 4
        assert stats["edge_count"] == 4
        assert stats["node_types"]["Module"] == 1
        assert stats["node_types"]["FunctionDef"] == 2
        assert stats["node_types"]["Assign"] == 1
        assert stats["edge_types"]["CHILD"] == 3
        assert stats["edge_types"]["CALLS"] == 1
        assert stats["has_source_info"] is True
