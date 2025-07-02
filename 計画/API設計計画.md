# API設計計画

## 概要
ast2graphの使いやすく拡張可能なAPIインターフェースの設計計画です。

## API設計原則

### 1. シンプルで直感的
- 最も一般的な使用方法は1-2行で実現
- 明確で一貫性のある命名規則
- 適切なデフォルト値

### 2. 段階的な複雑性
- 基本API: 初心者向けのシンプルな関数
- 中級API: カスタマイズ可能なクラス
- 上級API: 完全な制御が可能な低レベルAPI

### 3. 型安全性
- 完全な型アノテーション
- mypyでの厳密な型チェック
- 実行時の型検証（オプション）

## API階層

### 1. 高レベルAPI（簡単・便利）

#### 1.1 基本関数
```python
# ast2graph/__init__.py で公開
from ast2graph import parse_file, parse_code, parse_directory

# 単一ファイル解析
graph = parse_file("example.py")

# コード文字列解析
graph = parse_code("def hello(): pass")

# ディレクトリ一括解析
results = parse_directory("./src", pattern="**/*.py")
```

#### 1.2 エクスポート関数
```python
from ast2graph import export_to_json, export_to_cypher

# JSON形式でエクスポート
json_data = export_to_json(graph)

# Cypherクエリとしてエクスポート
cypher_query = export_to_cypher(graph)
```

### 2. 中レベルAPI（カスタマイズ可能）

#### 2.1 ASTProcessor クラス
```python
from ast2graph import ASTProcessor, ProcessingConfig

# カスタム設定で処理
config = ProcessingConfig(
    max_nodes_per_file=10000,
    include_docstrings=True,
    follow_imports=False
)

processor = ASTProcessor(config)
graph = processor.process_file("example.py")
```

#### 2.2 BatchProcessor クラス
```python
from ast2graph import BatchProcessor, ParallelConfig

# 並列バッチ処理
parallel_config = ParallelConfig(
    max_workers=8,
    chunk_size=100
)

batch_processor = BatchProcessor(
    processing_config=config,
    parallel_config=parallel_config
)

# プログレスコールバック付き
def on_progress(completed: int, total: int):
    print(f"Progress: {completed}/{total}")

results = batch_processor.process_files(
    file_paths,
    progress_callback=on_progress
)
```

### 3. 低レベルAPI（完全な制御）

#### 3.1 個別コンポーネント
```python
from ast2graph.parser import Parser
from ast2graph.graph_builder import GraphBuilder
from ast2graph.dependency_extractor import DependencyExtractor

# 手動でパイプラインを構築
parser = Parser()
ast_tree = parser.parse_source(source_code)

builder = GraphBuilder()
graph = builder.build_graph(ast_tree, source_id="custom_id")

extractor = DependencyExtractor()
graph = extractor.extract_dependencies(graph)
```

#### 3.2 カスタムビジター
```python
from ast2graph.visitors import BaseASTVisitor

class CustomVisitor(BaseASTVisitor):
    def visit_FunctionDef(self, node):
        # カスタム処理
        super().visit_FunctionDef(node)
        
    def visit_ClassDef(self, node):
        # カスタム処理
        super().visit_ClassDef(node)

# カスタムビジターを使用
builder = GraphBuilder(visitor_class=CustomVisitor)
```

## エラーハンドリング

### 1. 例外階層
```python
# ast2graph/exceptions.py

class AST2GraphError(Exception):
    """基底例外クラス"""

class ParseError(AST2GraphError):
    """構文解析エラー"""
    def __init__(self, message: str, line_no: int = None, file_path: str = None):
        self.line_no = line_no
        self.file_path = file_path
        super().__init__(message)

class GraphBuildError(AST2GraphError):
    """グラフ構築エラー"""

class ValidationError(AST2GraphError):
    """データ検証エラー"""

class NodeLimitExceeded(AST2GraphError):
    """ノード数制限超過"""
    
class MemoryLimitExceeded(AST2GraphError):
    """メモリ制限超過"""
```

### 2. エラー処理パターン
```python
# 厳密モード（デフォルト）
try:
    graph = parse_file("example.py")
except ParseError as e:
    print(f"Parse error at line {e.line_no}: {e}")
    
# 寛容モード
graph = parse_file("example.py", strict=False)
if graph.has_errors():
    for error in graph.errors:
        print(f"Warning: {error}")
```

## 設定とカスタマイズ

### 1. グローバル設定
```python
# ast2graph/config.py

from ast2graph import configure

configure(
    default_encoding="utf-8",
    max_memory_mb=1000,
    log_level="INFO"
)
```

### 2. コンテキストマネージャー
```python
from ast2graph import processing_context

with processing_context(max_nodes=100000, timeout=60):
    # この範囲内では設定が適用される
    graph = parse_file("large_file.py")
```

## プラグインシステム

### 1. フック機構
```python
from ast2graph import register_hook

@register_hook("pre_parse")
def validate_file(file_path: str) -> bool:
    """ファイル解析前の検証"""
    return file_path.endswith(".py")

@register_hook("post_build")
def enrich_graph(graph: GraphStructure) -> GraphStructure:
    """グラフ構築後の処理"""
    # カスタムメタデータ追加など
    return graph
```

### 2. カスタムエクスポーター
```python
from ast2graph import register_exporter

@register_exporter("graphml")
def export_to_graphml(graph: GraphStructure) -> str:
    """GraphML形式でエクスポート"""
    # 実装
    pass

# 使用
from ast2graph import export_graph
graphml_data = export_graph(graph, format="graphml")
```

## CLI インターフェース

### 基本コマンド
```bash
# 単一ファイル解析
ast2graph parse example.py

# ディレクトリ解析
ast2graph parse ./src --pattern "**/*.py" --output result.json

# バッチ処理
ast2graph batch file_list.txt --workers 8 --format cypher

# 統計情報表示
ast2graph stats ./src
```

### 設定ファイル
```yaml
# .ast2graph.yml
processing:
  max_nodes_per_file: 50000
  include_docstrings: true
  encoding: utf-8

parallel:
  max_workers: 8
  chunk_size: 50

export:
  format: json
  pretty_print: true
```

## バージョニングとの後方互換性

### 1. セマンティックバージョニング
- メジャー: 破壊的変更
- マイナー: 機能追加（後方互換）
- パッチ: バグ修正

### 2. 非推奨化ポリシー
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

### 3. 移行ガイド
- 各メジャーバージョンアップごとに移行ガイドを提供
- 自動移行スクリプトの提供（可能な場合）

## ドキュメント計画

### 1. APIリファレンス
- 全公開APIの詳細説明
- パラメータ、戻り値、例外
- 使用例

### 2. チュートリアル
- クイックスタート
- 一般的な使用例
- 高度な使用方法

### 3. レシピ集
- 特定のタスクの実現方法
- パフォーマンスチューニング
- トラブルシューティング

## 実装優先順位

### Phase 1: 基本API（Week 1）
1. 高レベル関数の実装
2. 基本的なエラーハンドリング
3. 最小限のドキュメント

### Phase 2: 中級API（Week 2）
1. カスタマイズ可能なクラス
2. 設定システム
3. CLIの基本実装

### Phase 3: 高度な機能（Week 3）
1. プラグインシステム
2. 高度なエラーハンドリング
3. 完全なドキュメント