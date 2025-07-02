"""Test module for dependency extraction functionality."""

import ast
from typing import List, Optional
import pytest

from ast2graph.dependency_extractor import (
    DependencyExtractor,
    ImportInfo,
    ReferenceInfo,
    ReferenceType,
)
from ast2graph.graph_structure import GraphStructure
from ast2graph.models import EdgeType


class TestImportInfo:
    """Test ImportInfo data class."""

    def test_import_info_creation(self):
        """Test creation of ImportInfo instance."""
        info = ImportInfo(
            module_name="os",
            alias=None,
            imported_names=[],
            is_relative=False,
            level=0
        )
        assert info.module_name == "os"
        assert info.alias is None
        assert info.imported_names == []
        assert info.is_relative is False
        assert info.level == 0

    def test_import_info_with_alias(self):
        """Test ImportInfo with alias."""
        info = ImportInfo(
            module_name="numpy",
            alias="np",
            imported_names=[],
            is_relative=False,
            level=0
        )
        assert info.module_name == "numpy"
        assert info.alias == "np"

    def test_import_info_with_specific_imports(self):
        """Test ImportInfo with specific imports."""
        info = ImportInfo(
            module_name="os.path",
            alias=None,
            imported_names=["join", "exists"],
            is_relative=False,
            level=0
        )
        assert info.module_name == "os.path"
        assert info.imported_names == ["join", "exists"]

    def test_relative_import_info(self):
        """Test relative ImportInfo."""
        info = ImportInfo(
            module_name="models",
            alias=None,
            imported_names=["User"],
            is_relative=True,
            level=1
        )
        assert info.module_name == "models"
        assert info.is_relative is True
        assert info.level == 1


class TestReferenceInfo:
    """Test ReferenceInfo data class."""

    def test_reference_info_creation(self):
        """Test creation of ReferenceInfo instance."""
        info = ReferenceInfo(
            name="print",
            reference_type=ReferenceType.FUNCTION_CALL,
            scope="module",
            line=10,
            column=4
        )
        assert info.name == "print"
        assert info.reference_type == ReferenceType.FUNCTION_CALL
        assert info.scope == "module"
        assert info.line == 10
        assert info.column == 4


class TestDependencyExtractor:
    """Test DependencyExtractor class."""

    def test_extractor_initialization(self):
        """Test DependencyExtractor initialization."""
        extractor = DependencyExtractor()
        assert extractor.imports == {}
        assert extractor.references == []
        assert extractor.scope_stack == ["module"]

    def test_extract_simple_import(self):
        """Test extraction of simple import statement."""
        code = "import os"
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        assert "os" in extractor.imports
        assert extractor.imports["os"].module_name == "os"
        assert extractor.imports["os"].alias is None
        assert extractor.imports["os"].is_relative is False

    def test_extract_import_with_alias(self):
        """Test extraction of import with alias."""
        code = "import numpy as np"
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        assert "np" in extractor.imports
        assert extractor.imports["np"].module_name == "numpy"
        assert extractor.imports["np"].alias == "np"

    def test_extract_from_import(self):
        """Test extraction of from...import statement."""
        code = "from os.path import join, exists"
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        assert "join" in extractor.imports
        assert "exists" in extractor.imports
        assert extractor.imports["join"].module_name == "os.path"
        assert extractor.imports["join"].imported_names == ["join"]
        assert extractor.imports["exists"].module_name == "os.path"
        assert extractor.imports["exists"].imported_names == ["exists"]

    def test_extract_relative_import(self):
        """Test extraction of relative import."""
        code = "from .models import User"
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        assert "User" in extractor.imports
        assert extractor.imports["User"].module_name == "models"
        assert extractor.imports["User"].is_relative is True
        assert extractor.imports["User"].level == 1

    def test_extract_function_call(self):
        """Test extraction of function call references."""
        code = """
import os
result = os.path.join('a', 'b')
"""
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        # Check that function call reference is captured
        function_refs = [r for r in extractor.references 
                        if r.reference_type == ReferenceType.FUNCTION_CALL]
        assert len(function_refs) > 0
        
        # Find the os.path.join reference
        join_ref = next((r for r in function_refs if "join" in r.name), None)
        assert join_ref is not None

    def test_extract_class_instantiation(self):
        """Test extraction of class instantiation."""
        code = """
class MyClass:
    pass

obj = MyClass()
"""
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        # Check that class instantiation is captured
        class_refs = [r for r in extractor.references 
                     if r.reference_type == ReferenceType.CLASS_INSTANTIATION]
        assert len(class_refs) == 1
        assert class_refs[0].name == "MyClass"

    def test_scope_tracking(self):
        """Test scope tracking during extraction."""
        code = """
def my_function():
    x = 1
    def inner():
        y = x
    return inner

class MyClass:
    def method(self):
        z = 2
"""
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        # Verify scope tracking worked correctly
        # This would be reflected in the references' scope information

    def test_wildcard_import(self):
        """Test extraction of wildcard import."""
        code = "from math import *"
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        # Check that wildcard import is properly recorded
        assert "*" in extractor.imports
        assert extractor.imports["*"].module_name == "math"

    def test_import_edges_in_graph(self):
        """Test that import dependencies create edges in the graph."""
        code = """
import os
from math import sqrt
"""
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        # First, we need to ensure the module node exists
        from ast2graph.models import ASTGraphNode
        module_node = ASTGraphNode(
            node_id="module_test",
            node_type="Module",
            label="test_module",
            ast_node_info={
                "name": "test_module",
                "source_file": "test.py"
            },
            source_location=(1, 0, 3, 0)
        )
        graph.add_node(module_node)
        module_node_id = module_node.node_id
        
        extractor.extract_dependencies(tree, graph, module_node_id=module_node_id)
        
        # Check that IMPORTS edges were created
        import_edges = graph.get_edges_of_type(EdgeType.IMPORTS)
        assert len(import_edges) >= 2  # One for os, one for math

    def test_reference_edges_in_graph(self):
        """Test that references create appropriate edges in the graph."""
        code = """
def foo():
    pass

def bar():
    foo()
"""
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        # Build the basic graph structure first
        from ast2graph.graph_builder import GraphBuilder
        from ast2graph.graph_structure import SourceInfo
        from datetime import datetime
        import hashlib
        
        # Create source info
        import uuid
        source_info = SourceInfo(
            source_id=str(uuid.uuid4()),
            file_path="test.py",
            file_hash=hashlib.sha256(code.encode()).hexdigest(),
            parsed_at=datetime.now(),
            line_count=6,
            size_bytes=len(code.encode())
        )
        
        builder = GraphBuilder(tree, source_info)
        graph = builder.build_graph()
        
        # Then extract dependencies
        extractor.extract_dependencies(tree, graph)
        
        # Check that USES edges were created
        uses_edges = graph.get_edges_of_type(EdgeType.USES)
        assert len(uses_edges) >= 1  # bar uses foo

    def test_complex_import_scenarios(self):
        """Test complex import scenarios."""
        code = """
import os.path
from typing import List, Dict, Optional
from ..models import User as UserModel
from . import utils
import sys, json
"""
        tree = ast.parse(code)
        
        extractor = DependencyExtractor()
        graph = GraphStructure()
        
        extractor.extract_dependencies(tree, graph)
        
        # Verify all imports were captured correctly
        assert "os.path" in extractor.imports
        assert "List" in extractor.imports
        assert "Dict" in extractor.imports
        assert "Optional" in extractor.imports
        assert "UserModel" in extractor.imports
        assert extractor.imports["UserModel"].module_name == "models"
        assert extractor.imports["UserModel"].alias == "UserModel"
        assert extractor.imports["UserModel"].level == 2
        assert "utils" in extractor.imports
        assert extractor.imports["utils"].level == 1
        assert "sys" in extractor.imports
        assert "json" in extractor.imports