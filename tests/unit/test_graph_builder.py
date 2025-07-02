"""Tests for graph_builder module."""

import ast
import uuid
from datetime import datetime
from typing import Any

import pytest

from ast2graph.exceptions import GraphBuildError
from ast2graph.graph_builder import GraphBuilder
from ast2graph.graph_structure import SourceInfo
from ast2graph.models import EdgeType


class TestGraphBuilder:
    """Test cases for GraphBuilder class."""

    @pytest.fixture
    def simple_source_info(self) -> SourceInfo:
        """Provide a simple SourceInfo instance."""
        return SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=1,
            size_bytes=17
        )

    @pytest.fixture
    def simple_ast(self) -> ast.AST:
        """Provide a simple AST for testing."""
        code = """
def hello():
    pass
"""
        return ast.parse(code)

    @pytest.fixture
    def complex_ast(self) -> ast.AST:
        """Provide a complex AST with multiple node types."""
        code = """
import os
from typing import List

class MyClass:
    def __init__(self):
        self.value = 42
    
    def method(self, arg: str) -> str:
        result = f"Hello {arg}"
        return result

def main():
    obj = MyClass()
    print(obj.method("world"))

if __name__ == "__main__":
    main()
"""
        return ast.parse(code)

    def test_graph_builder_initialization(self, simple_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test GraphBuilder initialization."""
        builder = GraphBuilder(simple_ast, simple_source_info)
        assert builder.ast_tree == simple_ast
        assert builder.source_info == simple_source_info
        assert builder.graph is not None
        assert builder._node_counter == 0
        assert builder._node_mapping == {}

    def test_build_graph_simple(self, simple_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test building graph from simple AST."""
        builder = GraphBuilder(simple_ast, simple_source_info)
        graph = builder.build_graph()
        
        # Check that nodes were created
        assert len(graph.nodes) > 0
        
        # Check Module node exists
        module_nodes = [n for n in graph.nodes.values() if n.node_type == "Module"]
        assert len(module_nodes) == 1
        
        # Check FunctionDef node exists
        func_nodes = [n for n in graph.nodes.values() if n.node_type == "FunctionDef"]
        assert len(func_nodes) == 1
        assert func_nodes[0].value == "hello"

    def test_build_graph_complex(self, complex_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test building graph from complex AST."""
        builder = GraphBuilder(complex_ast, simple_source_info)
        graph = builder.build_graph()
        
        # Check various node types
        node_types = {node.node_type for node in graph.nodes.values()}
        expected_types = {"Module", "Import", "ImportFrom", "ClassDef", "FunctionDef"}
        assert expected_types.issubset(node_types)
        
        # Check class and methods
        class_nodes = [n for n in graph.nodes.values() if n.node_type == "ClassDef"]
        assert len(class_nodes) == 1
        assert class_nodes[0].value == "MyClass"
        
        # Check functions
        func_nodes = [n for n in graph.nodes.values() if n.node_type == "FunctionDef"]
        func_names = {n.value for n in func_nodes}
        assert "__init__" in func_names
        assert "method" in func_names
        assert "main" in func_names

    def test_node_id_generation(self, simple_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test that node IDs are generated deterministically."""
        builder1 = GraphBuilder(simple_ast, simple_source_info)
        graph1 = builder1.build_graph()
        
        builder2 = GraphBuilder(simple_ast, simple_source_info)
        graph2 = builder2.build_graph()
        
        # Check that the same number of nodes are created
        assert len(graph1.nodes) == len(graph2.nodes)
        
        # Check that node IDs are valid UUIDs
        node_ids = list(graph1.nodes.keys())
        for node_id in node_ids:
            # Verify it's a valid UUID
            uuid.UUID(node_id)

    def test_edge_creation(self, simple_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test that edges are created correctly."""
        builder = GraphBuilder(simple_ast, simple_source_info)
        graph = builder.build_graph()
        
        # Check that edges exist
        assert len(graph.edges) > 0
        
        # Check Module -> FunctionDef edge
        module_node = next(n for n in graph.nodes.values() if n.node_type == "Module")
        func_node = next(n for n in graph.nodes.values() if n.node_type == "FunctionDef")
        
        # Find edge from Module to FunctionDef
        module_to_func_edges = [
            e for e in graph.edges 
            if e.source_id == module_node.id and e.target_id == func_node.id
        ]
        assert len(module_to_func_edges) == 1
        assert module_to_func_edges[0].edge_type == EdgeType.CHILD

    def test_deterministic_processing(self, complex_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test that the same AST produces the same graph."""
        graphs = []
        for _ in range(3):
            builder = GraphBuilder(complex_ast, simple_source_info)
            graphs.append(builder.build_graph())
        
        # Check all graphs have the same number of nodes and edges
        node_counts = [len(g.nodes) for g in graphs]
        edge_counts = [len(g.edges) for g in graphs]
        assert len(set(node_counts)) == 1
        assert len(set(edge_counts)) == 1
        
        # Check node types are consistent
        for i in range(1, len(graphs)):
            types1 = sorted([n.node_type for n in graphs[0].nodes.values()])
            types2 = sorted([n.node_type for n in graphs[i].nodes.values()])
            assert types1 == types2

    def test_empty_module(self, simple_source_info: SourceInfo) -> None:
        """Test handling of empty module."""
        empty_ast = ast.parse("")
        builder = GraphBuilder(empty_ast, simple_source_info)
        graph = builder.build_graph()
        
        # Should have at least a Module node
        assert len(graph.nodes) >= 1
        module_nodes = [n for n in graph.nodes.values() if n.node_type == "Module"]
        assert len(module_nodes) == 1

    def test_node_mapping(self, simple_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test that AST nodes are properly mapped to graph nodes."""
        builder = GraphBuilder(simple_ast, simple_source_info)
        graph = builder.build_graph()
        
        # Check that the mapping contains entries
        assert len(builder._node_mapping) > 0
        
        # Check that mapped nodes exist in the graph
        for ast_node, node_id in builder._node_mapping.items():
            assert node_id in graph.nodes
            graph_node = graph.nodes[node_id]
            assert graph_node.node_type == type(ast_node).__name__

    def test_source_info_propagation(self, simple_ast: ast.AST, simple_source_info: SourceInfo) -> None:
        """Test that source info is properly propagated to nodes."""
        builder = GraphBuilder(simple_ast, simple_source_info)
        graph = builder.build_graph()
        
        # Check that all nodes have the correct source_id
        for node in graph.nodes.values():
            assert node.source_id == simple_source_info.file_path

    def test_line_number_tracking(self) -> None:
        """Test that line numbers are correctly tracked."""
        code = """def func1():
    pass

def func2():
    pass"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Find function nodes
        func_nodes = [n for n in graph.nodes.values() if n.node_type == "FunctionDef"]
        func_nodes.sort(key=lambda n: n.lineno)
        
        assert len(func_nodes) == 2
        assert func_nodes[0].value == "func1"
        assert func_nodes[0].lineno == 1
        assert func_nodes[1].value == "func2"
        assert func_nodes[1].lineno == 4

    def test_error_handling(self, simple_source_info: SourceInfo) -> None:
        """Test error handling during graph building."""
        # Create a real AST but break it to cause errors
        code = "def test(): pass"
        real_ast = ast.parse(code)
        
        # Break the AST by setting invalid line number
        real_ast.body[0].lineno = -1  # This should cause validation error in ASTGraphNode
        
        builder = GraphBuilder(real_ast, simple_source_info)
        
        # Should handle the error gracefully
        with pytest.raises(GraphBuildError) as exc_info:
            builder.build_graph()
        
        # Verify the error message
        assert "Failed to build graph" in str(exc_info.value)

    def test_import_handling(self) -> None:
        """Test handling of import statements."""
        code = """import os
import sys as system
from typing import List, Dict
from collections.abc import Mapping"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Check Import nodes
        import_nodes = [n for n in graph.nodes.values() if n.node_type == "Import"]
        assert len(import_nodes) == 2
        
        # Check ImportFrom nodes
        import_from_nodes = [n for n in graph.nodes.values() if n.node_type == "ImportFrom"]
        assert len(import_from_nodes) == 2

    def test_class_with_inheritance(self) -> None:
        """Test handling of class with inheritance."""
        code = """class Base:
    pass

class Derived(Base):
    def method(self):
        pass"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Check ClassDef nodes
        class_nodes = [n for n in graph.nodes.values() if n.node_type == "ClassDef"]
        assert len(class_nodes) == 2
        
        class_names = {n.value for n in class_nodes}
        assert "Base" in class_names
        assert "Derived" in class_names

    def test_function_with_decorators(self) -> None:
        """Test handling of decorated functions."""
        code = """@property
@staticmethod
def decorated_func():
    pass"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Check FunctionDef node exists
        func_nodes = [n for n in graph.nodes.values() if n.node_type == "FunctionDef"]
        assert len(func_nodes) == 1
        assert func_nodes[0].value == "decorated_func"

    def test_async_function(self) -> None:
        """Test handling of async functions."""
        code = """async def async_func():
    await some_coroutine()"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Check AsyncFunctionDef node exists
        async_func_nodes = [n for n in graph.nodes.values() if n.node_type == "AsyncFunctionDef"]
        assert len(async_func_nodes) == 1
        assert async_func_nodes[0].value == "async_func"

    def test_assignment_handling(self) -> None:
        """Test handling of assignments."""
        code = """x = 42
y = x + 1
z = func(y)"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Check Assign nodes
        assign_nodes = [n for n in graph.nodes.values() if n.node_type == "Assign"]
        assert len(assign_nodes) == 3
        
        # Check Name nodes
        name_nodes = [n for n in graph.nodes.values() if n.node_type == "Name"]
        assert len(name_nodes) >= 5  # x, y, z as targets and x, y as references
        
        # Check that we have both DEFINES and REFERENCES edges
        edge_types = {e.edge_type for e in graph.edges}
        assert EdgeType.DEFINES in edge_types
        assert EdgeType.REFERENCES in edge_types

    def test_call_and_control_flow(self) -> None:
        """Test handling of function calls and control flow."""
        code = """def process(data):
    if data:
        result = transform(data)
        for item in result:
            print(item)
    else:
        return None
    return result"""
        
        source_info = SourceInfo(
            source_id="550e8400-e29b-41d4-a716-446655440000",
            file_path="test.py",
            file_hash="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=code.count('\n') + 1,
            size_bytes=len(code)
        )
        ast_tree = ast.parse(code)
        builder = GraphBuilder(ast_tree, source_info)
        graph = builder.build_graph()
        
        # Check control flow nodes
        if_nodes = [n for n in graph.nodes.values() if n.node_type == "If"]
        assert len(if_nodes) == 1
        assert if_nodes[0].value == "conditional"
        
        for_nodes = [n for n in graph.nodes.values() if n.node_type == "For"]
        assert len(for_nodes) == 1
        assert for_nodes[0].value == "loop"
        
        # Check Call nodes
        call_nodes = [n for n in graph.nodes.values() if n.node_type == "Call"]
        assert len(call_nodes) >= 2  # transform() and print()
        
        # Check Return nodes
        return_nodes = [n for n in graph.nodes.values() if n.node_type == "Return"]
        assert len(return_nodes) == 2
        assert all(n.value == "return" for n in return_nodes)