"""パフォーマンステスト.

処理速度、メモリ使用量、決定性を検証する。
"""
import gc
import hashlib
import json
import time
from pathlib import Path

from ast2graph import parse_code, parse_directory, parse_file, parse_files_stream


class TestProcessingSpeed:
    """処理速度のテスト."""

    def test_single_file_performance(self, tmp_path: Path) -> None:
        """単一ファイルの処理速度."""
        # Arrange - 中規模のファイル（1000行程度）
        lines = ['"""Performance test file."""']

        # 100個の関数
        for i in range(100):
            lines.extend([
                f"\ndef function_{i}(x, y, z):",
                f'    """Function {i} documentation."""',
                "    result = x * y + z",
                "    for j in range(10):",
                f"        result += j * {i}",
                "    return result",
            ])

        # 50個のクラス
        for i in range(50):
            lines.extend([
                f"\nclass Class_{i}:",
                f'    """Class {i} documentation."""',
                "    ",
                "    def __init__(self, value):",
                "        self.value = value",
                "    ",
                "    def method(self):",
                "        return self.value * 2",
            ])

        code = "\n".join(lines)
        file_path = tmp_path / "large_file.py"
        file_path.write_text(code)

        # Act - 処理時間を測定
        start_time = time.time()
        result = parse_file(str(file_path))
        end_time = time.time()

        processing_time = end_time - start_time

        # Assert
        assert result is not None
        assert processing_time < 1.0  # 1秒以内に処理完了

        # ノード数の確認
        nodes = result["nodes"]
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        class_nodes = [n for n in nodes if n["type"] == "ClassDef"]

        assert len(func_nodes) >= 150  # 100個の関数 + 50個のクラスメソッド
        assert len(class_nodes) == 50

    def test_directory_processing_speed(self, tmp_path: Path) -> None:
        """ディレクトリ処理速度（300ファイル目標）."""
        # Arrange - 300個のファイルを作成
        project = tmp_path / "large_project"
        project.mkdir()

        # モジュール構造を作成
        for i in range(10):  # 10個のパッケージ
            package = project / f"package_{i}"
            package.mkdir()
            (package / "__init__.py").write_text(f'"""Package {i}."""')

            for j in range(30):  # 各パッケージに30ファイル
                module_content = f'''
"""Module {i}.{j}."""

def function_a():
    return {i * j}

def function_b(x, y):
    return x + y + {i + j}

class Module{i}_{j}:
    def __init__(self):
        self.value = {i * 10 + j}

    def process(self):
        return self.value * 2
'''
                (package / f"module_{j}.py").write_text(module_content)

        # Act - 処理時間を測定
        start_time = time.time()
        results = parse_directory(str(project), recursive=True)
        end_time = time.time()

        processing_time = end_time - start_time
        files_processed = len(results)

        # Assert
        assert files_processed >= 300  # 300ファイル以上処理
        assert processing_time < 600  # 10分（600秒）以内

        # 1ファイルあたりの平均処理時間
        avg_time_per_file = processing_time / files_processed
        assert avg_time_per_file < 2.0  # 1ファイルあたり2秒以内

    def test_streaming_performance(self, tmp_path: Path) -> None:
        """ストリーミング処理のパフォーマンス."""
        # Arrange - 100個のファイル
        file_paths = []
        for i in range(100):
            content = f'''
def process_{i}(data):
    result = []
    for item in data:
        if item > {i}:
            result.append(item * 2)
    return result

class Processor_{i}:
    def __init__(self):
        self.threshold = {i}

    def run(self, data):
        return process_{i}(data)
'''
            file_path = tmp_path / f"stream_{i}.py"
            file_path.write_text(content)
            file_paths.append(str(file_path))

        # Act - ストリーミング処理時間
        start_time = time.time()
        processed_count = 0

        for file_path, result in parse_files_stream(file_paths):
            if result is not None:
                processed_count += 1

        end_time = time.time()
        streaming_time = end_time - start_time

        # Assert
        assert processed_count == 100
        assert streaming_time < 60  # 1分以内に100ファイル処理

    def test_complex_code_performance(self, tmp_path: Path) -> None:
        """複雑なコードの処理性能."""
        # Arrange - 深くネストした複雑なコード
        code = '''
"""Complex code with deep nesting and many branches."""

def complex_function(data, options=None):
    """Process data with complex logic."""
    if options is None:
        options = {}

    result = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            if "type" in item:
                if item["type"] == "A":
                    for key, value in item.items():
                        if key != "type":
                            if isinstance(value, list):
                                for v in value:
                                    if v > 0:
                                        result.append(v * 2)
                            elif isinstance(value, (int, float)):
                                if value > 0:
                                    result.append(value)
                elif item["type"] == "B":
                    try:
                        processed = item.get("data", [])
                        for p in processed:
                            if p is not None:
                                result.append(p)
                    except Exception as e:
                        pass
        elif isinstance(item, list):
            for sub_item in item:
                if sub_item:
                    result.extend(complex_function([sub_item], options))

    return result

class ComplexProcessor:
    """Complex data processor with multiple methods."""

    def __init__(self, config):
        self.config = config
        self._cache = {}
        self._initialized = False

    def initialize(self):
        """Initialize the processor."""
        if not self._initialized:
            for key, value in self.config.items():
                if isinstance(value, dict):
                    self._cache[key] = self._process_config(value)
            self._initialized = True

    def _process_config(self, config):
        """Process configuration recursively."""
        result = {}
        for k, v in config.items():
            if isinstance(v, dict):
                result[k] = self._process_config(v)
            else:
                result[k] = v
        return result

    def process(self, data):
        """Main processing method."""
        self.initialize()
        return [self._process_item(item) for item in data]

    def _process_item(self, item):
        """Process a single item."""
        if isinstance(item, dict):
            return {k: self._process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._process_item(i) for i in item]
        else:
            return item
'''

        # 上記のコードを10回繰り返して大きなファイルを作成
        large_code = "\n\n".join([code] * 10)
        file_path = tmp_path / "complex.py"
        file_path.write_text(large_code)

        # Act
        start_time = time.time()
        result = parse_file(str(file_path))
        end_time = time.time()

        processing_time = end_time - start_time

        # Assert
        assert result is not None
        assert processing_time < 5.0  # 5秒以内に処理

        # 複雑さの確認
        nodes = result["nodes"]
        assert len(nodes) > 1000  # 多数のノード


class TestMemoryUsage:
    """メモリ使用量のテスト."""

    def test_memory_efficiency_single_file(self, tmp_path: Path) -> None:
        """単一ファイル処理のメモリ効率."""
        # Arrange - メモリ使用量測定のための大きなファイル
        lines = []
        for i in range(1000):
            lines.append(f"variable_{i} = {i}")
            if i % 10 == 0:
                lines.extend([
                    f"def function_{i}():",
                    f"    return {i}",
                    ""
                ])

        code = "\n".join(lines)
        file_path = tmp_path / "memory_test.py"
        file_path.write_text(code)

        # Act - ガベージコレクションを実行してからメモリ測定
        gc.collect()

        # メモリ使用量の簡易測定（実際のメモリプロファイリングは別途必要）
        result = parse_file(str(file_path))

        # Assert
        assert result is not None

        # 結果のサイズを確認（JSONシリアライズ後）
        json_str = json.dumps(result)
        result_size_mb = len(json_str) / (1024 * 1024)

        assert result_size_mb < 10  # 結果サイズが10MB未満

    def test_streaming_memory_efficiency(self, tmp_path: Path) -> None:
        """ストリーミング処理のメモリ効率性."""
        # Arrange - 50個の中規模ファイル
        file_paths = []
        for i in range(50):
            lines = [f"# File {i}"]
            for j in range(100):
                lines.append(f"var_{i}_{j} = {i * j}")

            file_path = tmp_path / f"stream_mem_{i}.py"
            file_path.write_text("\n".join(lines))
            file_paths.append(str(file_path))

        # Act - ストリーミング処理（結果を保持しない）
        gc.collect()
        processed = 0
        max_nodes = 0

        for result_dict in parse_files_stream(file_paths):
            if "graph" in result_dict:
                result = result_dict["graph"]
                processed += 1
                max_nodes = max(max_nodes, result["metadata"]["total_nodes"])
                # 結果をすぐに破棄（メモリ効率的）
                del result

        gc.collect()

        # Assert
        assert processed == 50
        assert max_nodes > 0
        # メモリ使用量が累積しないことを確認（実際の測定は環境依存）

    def test_large_graph_memory_usage(self, tmp_path: Path) -> None:
        """大規模グラフ構造のメモリ使用量."""
        # Arrange - 多数のノードとエッジを持つコード
        code = '''
"""Large graph structure test."""
'''

        # 200個のクラスと相互参照
        for i in range(200):
            code += f'''
class Class{i}:
    def __init__(self):
        self.id = {i}
'''
            # 他のクラスへの参照
            if i > 0:
                code += f'''
    def use_previous(self):
        return Class{i-1}()
'''
            if i < 199:
                code += f'''
    def use_next(self):
        return Class{i+1}()
'''

        file_path = tmp_path / "large_graph.py"
        file_path.write_text(code)

        # Act
        gc.collect()
        result = parse_file(str(file_path), include_dependencies=True)

        # Assert
        assert result is not None

        # グラフのサイズ確認
        nodes = result["nodes"]
        edges = result["edges"]

        assert len(nodes) > 600  # 200クラス + メソッド
        assert len(edges) > 400  # 親子関係 + 依存関係

        # メモリ効率の確認（概算）
        json_size = len(json.dumps(result))
        size_per_node = json_size / len(nodes)

        assert size_per_node < 1000  # 1ノードあたり1KB未満


class TestDeterminism:
    """決定性（同一入力で同一出力）のテスト."""

    def test_consistent_node_ids(self, tmp_path: Path) -> None:
        """ノードIDの一貫性テスト."""
        # Arrange
        code = '''
def function1():
    return 1

def function2():
    return 2

class MyClass:
    def method1(self):
        return function1()

    def method2(self):
        return function2()
'''

        # Act - 同じコードを3回解析
        results = []
        for i in range(3):
            result = parse_code(code, filename=f"test_{i}.py")
            results.append(result)

        # Assert - ノードIDの一貫性を確認
        for i in range(1, 3):
            nodes1 = sorted(results[0]["nodes"], key=lambda n: n["id"])
            nodes2 = sorted(results[i]["nodes"], key=lambda n: n["id"])

            assert len(nodes1) == len(nodes2)

            # 同じ位置のノードが同じ型とプロパティを持つ
            for n1, n2 in zip(nodes1, nodes2, strict=False):
                assert n1["type"] == n2["type"]
                assert n1["properties"] == n2["properties"]

    def test_consistent_graph_structure(self, tmp_path: Path) -> None:
        """グラフ構造の一貫性テスト."""
        # Arrange
        code = '''
import math
from typing import List

def calculate(values: List[float]) -> float:
    """Calculate sum of squares."""
    result = 0.0
    for val in values:
        result += val ** 2
    return math.sqrt(result)

class Calculator:
    def __init__(self):
        self.history = []

    def compute(self, values: List[float]) -> float:
        result = calculate(values)
        self.history.append(result)
        return result
'''

        # Act - 複数回解析して結果を比較
        results = []
        for _ in range(5):
            result = parse_code(code, include_dependencies=True)
            results.append(result)

        # Assert - エッジの一貫性
        reference_edges = sorted(results[0]["edges"], key=lambda e: (e["source"], e["target"], e["type"]))

        for i in range(1, 5):
            current_edges = sorted(results[i]["edges"], key=lambda e: (e["source"], e["target"], e["type"]))
            assert len(reference_edges) == len(current_edges)

            # エッジタイプの分布が同じ
            ref_types = [e["type"] for e in reference_edges]
            cur_types = [e["type"] for e in current_edges]
            assert ref_types == cur_types

    def test_hash_based_determinism(self, tmp_path: Path) -> None:
        """ハッシュベースの決定性確認."""
        # Arrange - 複雑なコード
        code = '''
"""Complex module for hash testing."""

class A:
    def method_a(self): pass

class B(A):
    def method_b(self): pass

class C(B):
    def method_c(self): pass

def factory(type_name):
    types = {"A": A, "B": B, "C": C}
    return types.get(type_name)

# 使用例
instances = [factory(t)() for t in ["A", "B", "C"]]
'''

        # Act - 結果のハッシュを計算
        hashes = []
        for _i in range(10):
            result = parse_code(code)
            # 非決定的なフィールドを除外
            if "source_info" in result:
                # source_idとparsed_atは非決定的
                result["source_info"] = {
                    k: v for k, v in result["source_info"].items()
                    if k not in ["source_id", "parsed_at"]
                }
            # metadataのcreated_atとexport_idも非決定的
            if "metadata" in result:
                result["metadata"] = {
                    k: v for k, v in result["metadata"].items()
                    if k not in ["created_at", "export_id"]
                }
            # 結果を正規化してJSON文字列化
            json_str = json.dumps(result, sort_keys=True)
            hash_value = hashlib.sha256(json_str.encode()).hexdigest()
            hashes.append(hash_value)

        # Assert - すべてのハッシュが同一
        assert len(set(hashes)) == 1, "Results are not deterministic"

    def test_file_order_independence(self, tmp_path: Path) -> None:
        """ファイル処理順序の独立性テスト."""
        # Arrange - 3つのファイル
        files = {
            "a.py": "class A: pass",
            "b.py": "class B: pass",
            "c.py": "class C: pass"
        }

        for name, content in files.items():
            (tmp_path / name).write_text(content)

        # Act - 異なる順序で処理
        file_paths1 = [str(tmp_path / name) for name in ["a.py", "b.py", "c.py"]]
        file_paths2 = [str(tmp_path / name) for name in ["c.py", "a.py", "b.py"]]
        file_paths3 = [str(tmp_path / name) for name in ["b.py", "c.py", "a.py"]]

        results1 = list(parse_files_stream(file_paths1))
        results2 = list(parse_files_stream(file_paths2))
        results3 = list(parse_files_stream(file_paths3))

        # Assert - 各ファイルの結果が順序に依存しない
        for path in file_paths1:
            # 各結果セットから同じファイルの結果を取得
            r1 = next((r["graph"] for r in results1 if r["file_path"] == path and "graph" in r), None)
            r2 = next((r["graph"] for r in results2 if r["file_path"] == path and "graph" in r), None)
            r3 = next((r["graph"] for r in results3 if r["file_path"] == path and "graph" in r), None)

            assert r1 is not None and r2 is not None and r3 is not None

            # ノード数とエッジ数が同じ
            assert r1["metadata"]["total_nodes"] == r2["metadata"]["total_nodes"] == r3["metadata"]["total_nodes"]
            assert r1["metadata"]["total_edges"] == r2["metadata"]["total_edges"] == r3["metadata"]["total_edges"]


class TestPerformanceOptimization:
    """パフォーマンス最適化の効果測定."""

    def test_caching_effectiveness(self, tmp_path: Path) -> None:
        """キャッシング効果の測定（将来の最適化用）."""
        # Arrange
        code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i))
'''

        # Act - 同じコードを複数回解析
        times = []
        for _i in range(5):
            start = time.time()
            parse_code(code)
            end = time.time()
            times.append(end - start)

        # Assert
        # 現在はキャッシングなしだが、将来の最適化で
        # 2回目以降が速くなることを期待
        assert all(t < 1.0 for t in times)  # すべて1秒以内

    def test_incremental_parsing_potential(self, tmp_path: Path) -> None:
        """インクリメンタル解析の可能性テスト."""
        # Arrange - 基本ファイルと変更版
        base_code = '''
def function1():
    return 1

def function2():
    return 2
'''

        modified_code = '''
def function1():
    return 1

def function2():
    return 2

def function3():  # 新規追加
    return 3
'''

        # Act
        base_result = parse_code(base_code)
        modified_result = parse_code(modified_code)

        # Assert - 変更部分の特定（将来の最適化用）
        base_nodes = base_result["nodes"]
        modified_nodes = modified_result["nodes"]

        # 新規ノードの数
        new_nodes_count = len(modified_nodes) - len(base_nodes)
        assert new_nodes_count > 0  # 新しい関数が追加されている

    def test_parallel_processing_benefit(self, tmp_path: Path) -> None:
        """並列処理の効果測定（将来の実装用）."""
        # Arrange - 独立した複数ファイル
        files = []
        for i in range(20):
            code = f'''
def process_{i}(data):
    result = []
    for item in data:
        result.append(item * {i})
    return result
'''
            file_path = tmp_path / f"parallel_{i}.py"
            file_path.write_text(code)
            files.append(str(file_path))

        # Act - シーケンシャル処理時間
        start = time.time()
        sequential_results = []
        for file_path in files:
            result = parse_file(file_path)
            sequential_results.append(result)
        sequential_time = time.time() - start

        # Act - ストリーミング処理時間（現在は並列化なし）
        start = time.time()
        streaming_results = list(parse_files_stream(files))
        time.time() - start

        # Assert
        assert len(sequential_results) == len(streaming_results) == 20
        # 将来の並列化実装で streaming_time < sequential_time となることを期待
        assert sequential_time < 30  # 30秒以内に完了
