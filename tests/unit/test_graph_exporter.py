"""GraphExporterクラスのテスト。"""

import json
import os
import tempfile
import uuid
from datetime import datetime
from io import StringIO

import pytest

from ast2graph.exceptions import ExportError
from ast2graph.graph_exporter import GraphExporter
from ast2graph.graph_structure import GraphStructure, SourceInfo
from ast2graph.models import ASTGraphEdge, ASTGraphNode, EdgeType


class TestGraphExporter:
    """GraphExporterクラスのテスト。"""

    def setup_method(self):
        """各テストメソッドの前に実行される。"""
        # テスト用のグラフ構造を作成
        self.graph = GraphStructure()

        # ソース情報を設定
        self.graph.source_info = SourceInfo(
            source_id=str(uuid.uuid4()),
            file_path="/test/example.py",
            file_hash="a" * 64,  # 64文字の有効なSHA256ハッシュ形式
            parsed_at=datetime.now(),
            encoding="utf-8",
            line_count=50,
            size_bytes=1024
        )

        # テスト用ノードを追加
        self.node1 = ASTGraphNode(
            node_id="node_1",
            node_type="FunctionDef",
            label="test_function",
            ast_node_info={"name": "test_function", "lineno": 10},
            source_location=(10, 0, 15, 0),
            metadata={"complexity": 5}
        )
        self.node2 = ASTGraphNode(
            node_id="node_2",
            node_type="Call",
            label="print",
            ast_node_info={"func": "print", "lineno": 12},
            source_location=(12, 4, 12, 20),
            metadata={}
        )

        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)

        # テスト用エッジを追加
        self.edge1 = ASTGraphEdge(
            edge_id="edge_1",
            source_id="node_1",
            target_id="node_2",
            edge_type=EdgeType.CHILD,
            label="child",
            metadata={"order": 1}
        )
        self.graph.add_edge(self.edge1)

        self.exporter = GraphExporter(self.graph)

    def test_export_to_dict_basic(self):
        """export_to_dict()の基本的な動作をテスト。"""
        result = self.exporter.export_to_dict()

        # 基本構造の確認
        assert "version" in result
        assert result["version"] == "1.0.0"
        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result
        assert "source_info" in result

        # ノード数とエッジ数の確認
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_export_to_dict_nodes(self):
        """ノードのエクスポートが正しく行われるかテスト。"""
        result = self.exporter.export_to_dict()

        nodes = result["nodes"]
        node1_dict = next(n for n in nodes if n["id"] == "node_1")

        assert node1_dict["type"] == "FunctionDef"
        assert node1_dict["label"] == "test_function"
        assert node1_dict["ast_node_info"]["name"] == "test_function"
        assert node1_dict["source_location"] == (10, 0, 15, 0)
        assert node1_dict["metadata"]["complexity"] == 5

    def test_export_to_dict_edges(self):
        """エッジのエクスポートが正しく行われるかテスト。"""
        result = self.exporter.export_to_dict()

        edges = result["edges"]
        edge1_dict = edges[0]

        assert edge1_dict["id"] == "edge_1"
        assert edge1_dict["source"] == "node_1"
        assert edge1_dict["target"] == "node_2"
        assert edge1_dict["edge_type"] == "CHILD"
        assert edge1_dict["label"] == "child"
        assert edge1_dict["metadata"]["order"] == 1

    def test_export_to_dict_metadata(self):
        """メタデータのエクスポートが正しく行われるかテスト。"""
        result = self.exporter.export_to_dict()

        metadata = result["metadata"]
        assert metadata["node_count"] == 2
        assert metadata["edge_count"] == 1
        assert "created_at" in metadata
        assert "export_id" in metadata
        assert "CHILD" in metadata["edge_types"]

    def test_export_to_dict_source_info(self):
        """ソース情報のエクスポートが正しく行われるかテスト。"""
        result = self.exporter.export_to_dict()

        source_info = result["source_info"]
        assert source_info["file_path"] == "/test/example.py"
        assert source_info["encoding"] == "utf-8"
        assert source_info["file_hash"] == "a" * 64
        assert source_info["size_bytes"] == 1024
        assert source_info["line_count"] == 50
        assert "source_id" in source_info
        assert "parsed_at" in source_info

    def test_export_to_dict_without_metadata(self):
        """メタデータなしでのエクスポートをテスト。"""
        result = self.exporter.export_to_dict(include_metadata=False)

        assert "metadata" not in result
        assert "nodes" in result
        assert "edges" in result

    def test_export_to_dict_without_source_info(self):
        """ソース情報なしでのエクスポートをテスト。"""
        result = self.exporter.export_to_dict(include_source_info=False)

        assert "source_info" not in result
        assert "nodes" in result
        assert "edges" in result

    def test_export_to_json(self):
        """export_to_json()の動作をテスト。"""
        json_str = self.exporter.export_to_json()

        # JSON形式として有効か確認
        data = json.loads(json_str)
        assert data["version"] == "1.0.0"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_export_to_json_compressed(self):
        """圧縮形式（インデントなし）でのJSON出力をテスト。"""
        json_str = self.exporter.export_to_json(indent=None)

        # 改行が含まれていないことを確認
        assert '\n' not in json_str

        # JSON形式として有効か確認
        data = json.loads(json_str)
        assert data["version"] == "1.0.0"

    def test_export_to_file(self):
        """export_to_file()の動作をテスト。"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            self.exporter.export_to_file(temp_path)

            # ファイルが作成されたか確認
            assert os.path.exists(temp_path)

            # ファイル内容の確認
            with open(temp_path, encoding='utf-8') as f:
                data = json.load(f)

            assert data["version"] == "1.0.0"
            assert len(data["nodes"]) == 2
            assert len(data["edges"]) == 1

        finally:
            # クリーンアップ
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_to_file_error(self):
        """ファイル書き込みエラーのテスト。"""
        # 書き込み不可能なパス
        invalid_path = "/invalid/path/to/file.json"

        with pytest.raises(ExportError) as exc_info:
            self.exporter.export_to_file(invalid_path)

        assert "Failed to write to file" in str(exc_info.value)

    def test_export_to_stream(self):
        """export_to_stream()の動作をテスト。"""
        stream = StringIO()
        self.exporter.export_to_stream(stream)

        # ストリームの内容を取得
        stream.seek(0)
        json_str = stream.read()

        # JSON形式として有効か確認
        data = json.loads(json_str)
        assert data["version"] == "1.0.0"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_export_empty_graph(self):
        """空のグラフのエクスポートをテスト。"""
        empty_graph = GraphStructure()
        exporter = GraphExporter(empty_graph)

        result = exporter.export_to_dict()

        assert result["version"] == "1.0.0"
        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0
        assert result["metadata"]["node_count"] == 0
        assert result["metadata"]["edge_count"] == 0

    def test_export_deterministic(self):
        """同一グラフのエクスポートが決定的であることをテスト。"""
        # 同じグラフを2回エクスポート
        json1 = self.exporter.export_to_json(indent=None)
        json2 = self.exporter.export_to_json(indent=None)

        # メタデータの動的な部分を除外して比較
        data1 = json.loads(json1)
        data2 = json.loads(json2)

        # created_atとexport_idは毎回異なるので除外
        del data1["metadata"]["created_at"]
        del data1["metadata"]["export_id"]
        del data2["metadata"]["created_at"]
        del data2["metadata"]["export_id"]

        # それ以外は同一であるべき
        assert json.dumps(data1, sort_keys=True) == json.dumps(data2, sort_keys=True)

    def test_export_with_none_metadata(self):
        """メタデータがNoneのノード/エッジのエクスポートをテスト。"""
        # メタデータなしのノードを追加
        node3 = ASTGraphNode(
            node_id="node_3",
            node_type="Name",
            label="x",
            ast_node_info={"id": "x"},
            source_location=(20, 0, 20, 1),
            metadata=None
        )
        self.graph.add_node(node3)

        result = self.exporter.export_to_dict()

        node3_dict = next(n for n in result["nodes"] if n["id"] == "node_3")
        assert node3_dict["metadata"] == {}

    def test_export_large_graph_performance(self):
        """大規模グラフのエクスポート性能をテスト。"""
        # 1000ノード、2000エッジのグラフを作成
        large_graph = GraphStructure()

        for i in range(1000):
            node = ASTGraphNode(
                node_id=f"node_{i}",
                node_type="Name",
                label=f"var_{i}",
                ast_node_info={"id": f"var_{i}"},
                source_location=(i, 0, i, 10)
            )
            large_graph.add_node(node)

        for i in range(999):
            edge = ASTGraphEdge(
                edge_id=f"edge_{i}",
                source_id=f"node_{i}",
                target_id=f"node_{i+1}",
                edge_type=EdgeType.NEXT,
                label="follows"
            )
            large_graph.add_edge(edge)

        exporter = GraphExporter(large_graph)

        # ストリーミング出力のテスト
        stream = StringIO()
        exporter.export_to_stream(stream)

        # 出力が完了していることを確認
        stream.seek(0)
        data = json.loads(stream.read())
        assert len(data["nodes"]) == 1000
        assert len(data["edges"]) == 999
