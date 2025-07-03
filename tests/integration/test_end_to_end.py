"""エンドツーエンドの統合テスト.

単一ファイル処理、ディレクトリ処理、ストリーミング処理の
完全なワークフローを検証する。
"""
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ast2graph import parse_code, parse_directory, parse_file, parse_files_stream
from ast2graph.exceptions import AST2GraphError
from ast2graph.models import EdgeType


class TestSingleFileWorkflow:
    """単一ファイル処理のエンドツーエンドテスト."""

    def test_simple_file_complete_workflow(self, tmp_path: Path) -> None:
        """単純なPythonファイルの完全な処理フロー."""
        # Arrange
        code = '''
def greet(name: str) -> str:
    """挨拶を返す関数."""
    return f"Hello, {name}!"

class Person:
    """人物を表すクラス."""
    def __init__(self, name: str):
        self.name = name
    
    def say_hello(self) -> str:
        return greet(self.name)
'''
        file_path = tmp_path / "simple.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result

        # ノードの検証
        nodes = result["nodes"]
        assert len(nodes) > 0
        
        # 関数定義ノードの確認
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        assert len(func_nodes) == 3  # greet, __init__, say_hello
        assert any(n["properties"]["name"] == "greet" for n in func_nodes)
        assert any(n["properties"]["name"] == "say_hello" for n in func_nodes)
        
        # クラス定義ノードの確認
        class_nodes = [n for n in nodes if n["type"] == "ClassDef"]
        assert len(class_nodes) == 1
        assert class_nodes[0]["properties"]["name"] == "Person"

        # エッジの検証
        edges = result["edges"]
        assert len(edges) > 0
        
        # 親子関係の確認
        child_edges = [e for e in edges if e["type"] == EdgeType.CHILD.value]
        assert len(child_edges) > 0

        # メタデータの検証
        metadata = result["metadata"]
        assert metadata["source_file"] == str(file_path)
        assert metadata["total_nodes"] == len(nodes)
        assert metadata["total_edges"] == len(edges)

    def test_file_with_dependencies(self, tmp_path: Path) -> None:
        """依存関係を含むファイルの処理."""
        # Arrange
        code = '''
import os
from typing import List, Dict
from collections import defaultdict

def process_data(data: List[Dict]) -> Dict:
    """データを処理する関数."""
    result = defaultdict(list)
    for item in data:
        key = item.get("type", "unknown")
        result[key].append(item)
    return dict(result)

class FileProcessor:
    """ファイル処理クラス."""
    def __init__(self, base_path: str):
        self.base_path = os.path.abspath(base_path)
    
    def process(self, filename: str) -> Dict:
        path = os.path.join(self.base_path, filename)
        # ファイル処理のロジック
        return {"path": path}
'''
        file_path = tmp_path / "dependencies.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path), include_dependencies=True)

        # Assert
        nodes = result["nodes"]
        edges = result["edges"]
        
        # import文の依存関係確認
        import_edges = [e for e in edges if e["type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) > 0
        
        # 使用関係の確認
        uses_edges = [e for e in edges if e["type"] == EdgeType.USES.value]
        assert len(uses_edges) > 0  # os.path.abspath, os.path.join等の使用

    def test_export_formats(self, tmp_path: Path) -> None:
        """様々なエクスポート形式のテスト."""
        # Arrange
        code = 'x = 1\ny = x + 2'
        
        # Act & Assert - dict形式（デフォルト）
        result_dict = parse_code(code)
        assert isinstance(result_dict, dict)
        assert "nodes" in result_dict
        
        # Act & Assert - JSON文字列形式
        result_json = parse_code(code, output_format="json")
        assert isinstance(result_json, str)
        parsed = json.loads(result_json)
        assert "nodes" in parsed
        
        # Act & Assert - GraphStructure形式
        result_graph = parse_code(code, output_format="graph")
        from ast2graph.graph_structure import GraphStructure
        assert isinstance(result_graph, GraphStructure)
        assert len(result_graph.nodes) > 0


class TestDirectoryWorkflow:
    """ディレクトリ処理のエンドツーエンドテスト."""

    def test_multi_file_project(self, tmp_path: Path) -> None:
        """複数ファイルプロジェクトの処理."""
        # Arrange - プロジェクト構造の作成
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        
        # __init__.py
        (project_dir / "__init__.py").write_text('"""My project."""\n__version__ = "1.0.0"')
        
        # models.py
        (project_dir / "models.py").write_text('''
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price
''')
        
        # services.py
        (project_dir / "services.py").write_text('''
from .models import User, Product

class UserService:
    def create_user(self, name: str, email: str) -> User:
        return User(name, email)

class ProductService:
    def create_product(self, name: str, price: float) -> Product:
        return Product(name, price)
''')
        
        # main.py
        (project_dir / "main.py").write_text('''
from .services import UserService, ProductService

def main():
    user_service = UserService()
    product_service = ProductService()
    
    user = user_service.create_user("Alice", "alice@example.com")
    product = product_service.create_product("Book", 29.99)
    
    print(f"User: {user.name}")
    print(f"Product: {product.name} - ${product.price}")

if __name__ == "__main__":
    main()
''')

        # Act
        results = parse_directory(str(project_dir), include_dependencies=True)

        # Assert
        assert isinstance(results, dict)
        assert len(results) == 4  # 4つのファイル
        
        # 各ファイルの結果を確認
        assert str(project_dir / "__init__.py") in results
        assert str(project_dir / "models.py") in results
        assert str(project_dir / "services.py") in results
        assert str(project_dir / "main.py") in results
        
        # models.pyの内容確認
        models_result = results[str(project_dir / "models.py")]
        models_nodes = models_result["nodes"]
        class_nodes = [n for n in models_nodes if n["type"] == "ClassDef"]
        assert len(class_nodes) == 2  # User, Product
        
        # services.pyの依存関係確認
        services_result = results[str(project_dir / "services.py")]
        services_edges = services_result["edges"]
        import_edges = [e for e in services_edges if e["type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) > 0  # models.pyからのインポート

    def test_nested_directory_structure(self, tmp_path: Path) -> None:
        """ネストしたディレクトリ構造の処理."""
        # Arrange
        project = tmp_path / "nested_project"
        (project / "src" / "core").mkdir(parents=True)
        (project / "src" / "utils").mkdir(parents=True)
        (project / "tests").mkdir()
        
        # ファイル作成
        (project / "src" / "__init__.py").write_text("")
        (project / "src" / "core" / "__init__.py").write_text("")
        (project / "src" / "core" / "engine.py").write_text("class Engine: pass")
        (project / "src" / "utils" / "__init__.py").write_text("")
        (project / "src" / "utils" / "helpers.py").write_text("def helper(): pass")
        (project / "tests" / "test_engine.py").write_text('''
from src.core.engine import Engine
def test_engine():
    engine = Engine()
    assert engine is not None
''')

        # Act
        results = parse_directory(str(project), recursive=True)

        # Assert
        assert len(results) == 6  # 全ファイル数
        
        # パスの確認
        expected_files = [
            "src/__init__.py",
            "src/core/__init__.py",
            "src/core/engine.py",
            "src/utils/__init__.py",
            "src/utils/helpers.py",
            "tests/test_engine.py"
        ]
        for expected in expected_files:
            full_path = str(project / expected)
            assert full_path in results, f"{expected} not found in results"


class TestStreamingWorkflow:
    """ストリーミング処理のエンドツーエンドテスト."""

    def test_streaming_large_file_set(self, tmp_path: Path) -> None:
        """大量ファイルのストリーミング処理."""
        # Arrange - 50個のファイルを作成
        for i in range(50):
            file_path = tmp_path / f"module_{i:03d}.py"
            content = f'''
def function_{i}(x):
    """Function {i}."""
    return x * {i}

class Class_{i}:
    """Class {i}."""
    def method(self):
        return function_{i}(42)
'''
            file_path.write_text(content)

        # Act - ストリーミング処理
        file_paths = [str(tmp_path / f"module_{i:03d}.py") for i in range(50)]
        results_count = 0
        total_nodes = 0
        total_edges = 0
        
        for result_dict in parse_files_stream(file_paths):
            results_count += 1
            if "graph" in result_dict:
                result = result_dict["graph"]
                total_nodes += result["metadata"]["total_nodes"]
                total_edges += result["metadata"]["total_edges"]

        # Assert
        assert results_count == 50
        assert total_nodes > 0
        assert total_edges > 0

    def test_streaming_memory_efficiency(self, tmp_path: Path) -> None:
        """ストリーミング処理のメモリ効率性確認."""
        # Arrange - 大きめのファイルを複数作成
        large_code = '''
# 大きなファイルのシミュレーション
''' + '\n'.join([f'var_{i} = {i}' for i in range(1000)])
        
        file_paths = []
        for i in range(10):
            file_path = tmp_path / f"large_{i}.py"
            file_path.write_text(large_code)
            file_paths.append(str(file_path))

        # Act - ストリーミング処理（結果を保持しない）
        processed = 0
        for result_dict in parse_files_stream(file_paths):
            if "graph" in result_dict:
                result = result_dict["graph"]
                processed += 1
                # 結果をすぐに破棄（メモリ効率的）
                assert "nodes" in result
                assert len(result["nodes"]) > 1000

        # Assert
        assert processed == 10

    def test_streaming_with_errors(self, tmp_path: Path) -> None:
        """エラーを含むファイルのストリーミング処理."""
        # Arrange
        files = {
            "valid.py": "x = 1",
            "syntax_error.py": "def invalid syntax",
            "valid2.py": "y = 2",
            "encoding_error.py": b"\xff\xfe invalid encoding",
            "valid3.py": "z = 3"
        }
        
        file_paths = []
        for name, content in files.items():
            file_path = tmp_path / name
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)
            file_paths.append(str(file_path))

        # Act
        results = list(parse_files_stream(file_paths))

        # Assert
        assert len(results) == 5  # 全ファイル分の結果
        
        # 正常なファイルは処理されている
        valid_results = [r for r in results if "graph" in r]
        assert len(valid_results) == 3  # valid.py, valid2.py, valid3.py
        
        # エラーファイルはerrorキーを持つ
        error_results = [r for r in results if "error" in r]
        assert len(error_results) == 2  # syntax_error.py, encoding_error.py


class TestCompleteWorkflowIntegration:
    """完全なワークフロー統合テスト."""

    def test_realistic_workflow(self, tmp_path: Path) -> None:
        """現実的な使用シナリオのテスト."""
        # Arrange - リアルなプロジェクト構造
        project = tmp_path / "real_project"
        project.mkdir()
        
        # config.py
        (project / "config.py").write_text('''
import os
from typing import Dict, Any

class Config:
    """アプリケーション設定."""
    def __init__(self):
        self.debug = os.environ.get("DEBUG", "False") == "True"
        self.database_url = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "database_url": self.database_url
        }
''')
        
        # app.py
        (project / "app.py").write_text('''
from config import Config
from typing import Optional

class Application:
    """メインアプリケーションクラス."""
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._initialized = False
    
    def initialize(self) -> None:
        """アプリケーションの初期化."""
        if self._initialized:
            return
        
        # 初期化処理
        print(f"Initializing with config: {self.config.to_dict()}")
        self._initialized = True
    
    def run(self) -> None:
        """アプリケーションの実行."""
        self.initialize()
        print("Application is running...")
''')
        
        # main.py
        (project / "main.py").write_text('''
#!/usr/bin/env python3
"""エントリーポイント."""
from app import Application

def main():
    """メイン関数."""
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
''')

        # Act - 様々な方法で解析
        # 1. 個別ファイル解析
        config_result = parse_file(str(project / "config.py"), include_dependencies=True)
        
        # 2. ディレクトリ全体解析
        all_results = parse_directory(str(project), include_dependencies=True)
        
        # 3. 特定ファイルのコード解析
        with open(project / "main.py") as f:
            main_code = f.read()
        main_result = parse_code(main_code, filename="main.py", include_dependencies=True)

        # Assert
        # config.pyの検証
        assert config_result is not None
        config_nodes = config_result["nodes"]
        config_classes = [n for n in config_nodes if n["type"] == "ClassDef"]
        assert len(config_classes) == 1
        assert config_classes[0]["properties"]["name"] == "Config"
        
        # ディレクトリ解析の検証
        assert len(all_results) == 3
        assert all(path in all_results for path in [
            str(project / "config.py"),
            str(project / "app.py"),
            str(project / "main.py")
        ])
        
        # 依存関係の検証
        app_result = all_results[str(project / "app.py")]
        app_edges = app_result["edges"]
        import_edges = [e for e in app_edges if e["type"] == EdgeType.IMPORTS.value]
        assert any(e for e in import_edges if "Config" in str(e))
        
        # main.pyの検証
        assert main_result is not None
        main_nodes = main_result["nodes"]
        func_nodes = [n for n in main_nodes if n["type"] == "FunctionDef"]
        assert any(n["properties"]["name"] == "main" for n in func_nodes)