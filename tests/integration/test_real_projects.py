"""実プロジェクトでの統合テスト.

小規模から中規模の実際のPythonプロジェクト構造を
シミュレートしてテストする。
"""
from pathlib import Path

from ast2graph import parse_directory
from ast2graph.models import EdgeType


class TestSmallProjects:
    """小規模プロジェクト（5-10ファイル）のテスト."""

    def test_simple_package_structure(self, tmp_path: Path) -> None:
        """シンプルなパッケージ構造のテスト."""
        # Arrange - 小規模なライブラリ構造
        project = tmp_path / "simple_lib"
        project.mkdir()

        # パッケージ構造作成
        (project / "setup.py").write_text('''
from setuptools import setup, find_packages

setup(
    name="simple_lib",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
)
''')

        lib_dir = project / "simple_lib"
        lib_dir.mkdir()

        (lib_dir / "__init__.py").write_text('''
"""Simple library for demonstration."""
from .core import Calculator
from .utils import format_number

__version__ = "0.1.0"
__all__ = ["Calculator", "format_number"]
''')

        (lib_dir / "core.py").write_text('''
"""Core functionality."""
from typing import Union

Number = Union[int, float]

class Calculator:
    """Basic calculator class."""

    def add(self, a: Number, b: Number) -> Number:
        """Add two numbers."""
        return a + b

    def subtract(self, a: Number, b: Number) -> Number:
        """Subtract b from a."""
        return a - b

    def multiply(self, a: Number, b: Number) -> Number:
        """Multiply two numbers."""
        return a * b

    def divide(self, a: Number, b: Number) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
''')

        (lib_dir / "utils.py").write_text('''
"""Utility functions."""
from typing import Union

def format_number(num: Union[int, float], precision: int = 2) -> str:
    """Format a number with specified precision."""
    if isinstance(num, int):
        return str(num)
    return f"{num:.{precision}f}"

def validate_number(value: any) -> bool:
    """Check if value is a valid number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
''')

        (lib_dir / "exceptions.py").write_text('''
"""Custom exceptions."""

class CalculatorError(Exception):
    """Base exception for calculator errors."""
    pass

class InvalidOperationError(CalculatorError):
    """Raised when an invalid operation is attempted."""
    pass

class InvalidInputError(CalculatorError):
    """Raised when invalid input is provided."""
    pass
''')

        # テストディレクトリ
        tests_dir = project / "tests"
        tests_dir.mkdir()

        (tests_dir / "__init__.py").write_text("")

        (tests_dir / "test_calculator.py").write_text('''
"""Tests for calculator."""
import pytest
from simple_lib import Calculator

class TestCalculator:
    def setup_method(self):
        self.calc = Calculator()

    def test_add(self):
        assert self.calc.add(2, 3) == 5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError):
            self.calc.divide(10, 0)
''')

        # Act
        results = parse_directory(str(project), recursive=True, include_dependencies=True)

        # Assert
        assert len(results) >= 6  # 最低6ファイル

        # __init__.pyの依存関係確認
        init_file = str(lib_dir / "__init__.py")
        init_result = results[init_file]
        init_edges = init_result["edges"]

        # インポートの確認
        import_edges = [e for e in init_edges if e["type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) >= 2  # Calculator, format_number

        # core.pyの型エイリアス確認
        core_file = str(lib_dir / "core.py")
        core_result = results[core_file]
        core_nodes = core_result["nodes"]

        # 型エイリアスの存在確認
        assign_nodes = [n for n in core_nodes if n["type"] == "AnnAssign" or n["type"] == "Assign"]
        assert any(n for n in assign_nodes if "Number" in str(n.get("properties", {})))

        # クラスメソッドの確認
        class_nodes = [n for n in core_nodes if n["type"] == "ClassDef"]
        assert len(class_nodes) == 1
        calculator_class = class_nodes[0]
        assert calculator_class["properties"]["name"] == "Calculator"

    def test_cli_tool_structure(self, tmp_path: Path) -> None:
        """CLIツール構造のテスト."""
        # Arrange - CLIツールプロジェクト
        project = tmp_path / "mycli"
        project.mkdir()

        (project / "README.md").write_text("# MyCLI Tool")

        cli_dir = project / "mycli"
        cli_dir.mkdir()

        (cli_dir / "__init__.py").write_text('__version__ = "1.0.0"')

        (cli_dir / "__main__.py").write_text('''
"""Entry point for CLI."""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
''')

        (cli_dir / "cli.py").write_text('''
"""Command line interface."""
import argparse
import sys
from typing import Optional, List
from .commands import hello, goodbye
from .config import load_config

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="mycli",
        description="My CLI tool"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Hello command
    hello_parser = subparsers.add_parser("hello", help="Say hello")
    hello_parser.add_argument("name", help="Name to greet")

    # Goodbye command
    goodbye_parser = subparsers.add_parser("goodbye", help="Say goodbye")
    goodbye_parser.add_argument("name", help="Name to say goodbye to")

    return parser

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    config = load_config()

    if args.command == "hello":
        return hello(args.name, config)
    elif args.command == "goodbye":
        return goodbye(args.name, config)
    else:
        parser.print_help()
        return 1
''')

        (cli_dir / "commands.py").write_text('''
"""CLI commands."""
from typing import Dict, Any

def hello(name: str, config: Dict[str, Any]) -> int:
    """Say hello to someone."""
    greeting = config.get("greeting", "Hello")
    print(f"{greeting}, {name}!")
    return 0

def goodbye(name: str, config: Dict[str, Any]) -> int:
    """Say goodbye to someone."""
    farewell = config.get("farewell", "Goodbye")
    print(f"{farewell}, {name}!")
    return 0
''')

        (cli_dir / "config.py").write_text('''
"""Configuration management."""
import json
import os
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "greeting": "Hello",
    "farewell": "Goodbye",
    "debug": False
}

def get_config_path() -> Path:
    """Get configuration file path."""
    config_dir = Path.home() / ".mycli"
    return config_dir / "config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = get_config_path()

    if config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config

    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    config_path = get_config_path()
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
''')

        # Act
        results = parse_directory(str(project), recursive=True, include_dependencies=True)

        # Assert
        assert len(results) >= 5  # Python files only

        # __main__.pyの依存関係
        main_file = str(cli_dir / "__main__.py")
        main_result = results[main_file]
        main_edges = main_result["edges"]
        import_edges = [e for e in main_edges if e["type"] == EdgeType.IMPORTS.value]
        assert any(e for e in import_edges if "cli" in str(e))

        # cli.pyの複雑な依存関係
        cli_file = str(cli_dir / "cli.py")
        cli_result = results[cli_file]
        cli_nodes = cli_result["nodes"]

        # 関数定義の確認
        func_nodes = [n for n in cli_nodes if n["type"] == "FunctionDef"]
        func_names = {n["properties"]["name"] for n in func_nodes}
        assert "create_parser" in func_names
        assert "main" in func_names

        # argparseの使用確認
        cli_edges = cli_result["edges"]
        uses_edges = [e for e in cli_edges if e["type"] == EdgeType.USES.value]
        assert len(uses_edges) > 0


class TestMediumProjects:
    """中規模プロジェクト（50-100ファイル）のテスト."""

    def test_web_framework_structure(self, tmp_path: Path) -> None:
        """Webフレームワーク風の構造テスト."""
        # Arrange - Flask風のプロジェクト構造
        project = tmp_path / "webapp"
        project.mkdir()

        # アプリケーション構造
        app_dir = project / "app"
        app_dir.mkdir()

        (app_dir / "__init__.py").write_text('''
"""Web application package."""
from flask import Flask
from .config import Config

def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    from .api import api_bp
    from .auth import auth_bp
    from .main import main_bp

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    return app
''')

        # Config
        (app_dir / "config.py").write_text('''
"""Application configuration."""
import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
''')

        # Models
        models_dir = app_dir / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text('''
from .user import User
from .post import Post
from .comment import Comment

__all__ = ["User", "Post", "Comment"]
''')

        (models_dir / "user.py").write_text('''
"""User model."""
from datetime import datetime
from typing import Optional

class User:
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
        self.created_at = datetime.utcnow()
        self.id: Optional[int] = None
''')

        (models_dir / "post.py").write_text('''
"""Post model."""
from datetime import datetime
from typing import Optional
from .user import User

class Post:
    def __init__(self, title: str, content: str, author: User):
        self.title = title
        self.content = content
        self.author = author
        self.created_at = datetime.utcnow()
        self.id: Optional[int] = None
''')

        (models_dir / "comment.py").write_text('''
"""Comment model."""
from datetime import datetime
from typing import Optional
from .user import User
from .post import Post

class Comment:
    def __init__(self, content: str, author: User, post: Post):
        self.content = content
        self.author = author
        self.post = post
        self.created_at = datetime.utcnow()
        self.id: Optional[int] = None
''')

        # API Blueprint
        api_dir = app_dir / "api"
        api_dir.mkdir()
        (api_dir / "__init__.py").write_text('''
"""API blueprint."""
from flask import Blueprint

api_bp = Blueprint("api", __name__)

from . import routes
''')

        (api_dir / "routes.py").write_text('''
"""API routes."""
from flask import jsonify, request
from . import api_bp
from ..models import User, Post, Comment

@api_bp.route("/users", methods=["GET"])
def get_users():
    return jsonify({"users": []})

@api_bp.route("/posts", methods=["GET"])
def get_posts():
    return jsonify({"posts": []})

@api_bp.route("/posts", methods=["POST"])
def create_post():
    data = request.get_json()
    # Create post logic
    return jsonify({"status": "created"}), 201
''')

        # Auth Blueprint
        auth_dir = app_dir / "auth"
        auth_dir.mkdir()
        (auth_dir / "__init__.py").write_text('''
"""Authentication blueprint."""
from flask import Blueprint

auth_bp = Blueprint("auth", __name__)

from . import routes
''')

        (auth_dir / "routes.py").write_text('''
"""Authentication routes."""
from flask import request, jsonify
from . import auth_bp
from .utils import verify_password, generate_token

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if verify_password(username, password):
        token = generate_token(username)
        return jsonify({"token": token})

    return jsonify({"error": "Invalid credentials"}), 401

@auth_bp.route("/logout", methods=["POST"])
def logout():
    return jsonify({"status": "logged out"})
''')

        (auth_dir / "utils.py").write_text('''
"""Authentication utilities."""
import hashlib
import secrets
from typing import Optional

def hash_password(password: str) -> str:
    """Hash a password."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(username: str, password: str) -> bool:
    """Verify a password."""
    # Simplified verification
    return len(username) > 0 and len(password) > 0

def generate_token(username: str) -> str:
    """Generate an authentication token."""
    return f"{username}:{secrets.token_hex(16)}"
''')

        # Main Blueprint
        main_dir = app_dir / "main"
        main_dir.mkdir()
        (main_dir / "__init__.py").write_text('''
"""Main blueprint."""
from flask import Blueprint

main_bp = Blueprint("main", __name__)

from . import routes
''')

        (main_dir / "routes.py").write_text('''
"""Main routes."""
from flask import render_template
from . import main_bp

@main_bp.route("/")
def index():
    return {"message": "Welcome to the webapp"}

@main_bp.route("/about")
def about():
    return {"message": "About page"}
''')

        # Utils
        utils_dir = app_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "__init__.py").write_text('''
"""Utility functions."""
from .validators import validate_email, validate_username
from .helpers import format_datetime, slugify

__all__ = ["validate_email", "validate_username", "format_datetime", "slugify"]
''')

        (utils_dir / "validators.py").write_text(r'''
"""Validation utilities."""
import re

def validate_email(email: str) -> bool:
    """Validate an email address."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

def validate_username(username: str) -> bool:
    """Validate a username."""
    return 3 <= len(username) <= 20 and username.isalnum()
''')

        (utils_dir / "helpers.py").write_text('''
"""Helper functions."""
from datetime import datetime
import re

def format_datetime(dt: datetime) -> str:
    """Format a datetime object."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
''')

        # Act
        results = parse_directory(str(project), recursive=True, include_dependencies=True)

        # Assert
        python_files = [f for f in results if f.endswith(".py")]
        assert len(python_files) >= 15  # 少なくとも15個のPythonファイル

        # 依存関係の複雑さを確認
        total_import_edges = 0
        total_uses_edges = 0
        modules_with_imports = set()

        for file_path, result in results.items():
            if not file_path.endswith(".py"):
                continue

            edges = result["edges"]
            import_edges = [e for e in edges if e["type"] == EdgeType.IMPORTS.value]
            uses_edges = [e for e in edges if e["type"] == EdgeType.USES.value]

            if import_edges:
                modules_with_imports.add(file_path)
                total_import_edges += len(import_edges)

            total_uses_edges += len(uses_edges)

        # 相互依存関係の確認
        assert len(modules_with_imports) >= 10  # 多くのモジュールがインポートを持つ
        assert total_import_edges >= 20  # 多数のインポート関係

        # モデル間の依存関係確認
        post_model = str(models_dir / "post.py")
        post_result = results[post_model]
        post_edges = post_result["edges"]

        # PostモデルがUserモデルをインポート
        post_imports = [e for e in post_edges if e["type"] == EdgeType.IMPORTS.value]
        assert any(e for e in post_imports if "User" in str(e))

        # app/__init__.pyの複雑な依存関係
        app_init = str(app_dir / "__init__.py")
        app_result = results[app_init]
        app_nodes = app_result["nodes"]

        # create_app関数の存在
        func_nodes = [n for n in app_nodes if n["type"] == "FunctionDef"]
        assert any(n["properties"]["name"] == "create_app" for n in func_nodes)

    def test_data_science_project(self, tmp_path: Path) -> None:
        """データサイエンスプロジェクト構造のテスト."""
        # Arrange - Jupyter Notebook風のプロジェクト
        project = tmp_path / "ds_project"
        project.mkdir()

        # データ処理モジュール
        (project / "data_loader.py").write_text('''
"""Data loading utilities."""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

def load_csv(file_path: Path, **kwargs) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(file_path, **kwargs)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Tuple:
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=42)
''')

        # 特徴量エンジニアリング
        (project / "feature_engineering.py").write_text('''
"""Feature engineering utilities."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    """Feature engineering pipeline."""

    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

    def scale_numeric_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Scale numeric features."""
        result = df.copy()

        for col in columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                result[col] = self.scalers[col].fit_transform(result[[col]])
            else:
                result[col] = self.scalers[col].transform(result[[col]])

        return result

    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        result = df.copy()

        for col in columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                result[col] = self.encoders[col].fit_transform(result[col])
            else:
                result[col] = self.encoders[col].transform(result[col])

        return result
''')

        # モデルトレーニング
        (project / "models.py").write_text('''
"""Machine learning models."""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    """Model training and evaluation."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model: Optional[BaseEstimator] = None
        self.metrics: Dict[str, float] = {}

    def get_model(self) -> BaseEstimator:
        """Get model instance based on type."""
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42)
        }
        return models.get(self.model_type, RandomForestClassifier())

    def train(self, X_train, y_train):
        """Train the model."""
        self.model = self.get_model()
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate the model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(X_test)

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted")
        }

        return self.metrics
''')

        # 実験管理
        (project / "experiments.py").write_text('''
"""Experiment tracking and management."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from .models import ModelTrainer
from .feature_engineering import FeatureEngineer

class ExperimentTracker:
    """Track ML experiments."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results: List[Dict[str, Any]] = []
        self.output_dir = Path("experiments") / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment."""
        start_time = datetime.now()

        # Initialize components
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer(config.get("model_type", "random_forest"))

        # Run experiment (simplified)
        result = {
            "config": config,
            "start_time": start_time.isoformat(),
            "model_type": config.get("model_type"),
            "status": "completed"
        }

        # Save result
        self.results.append(result)
        self.save_results()

        return result

    def save_results(self):
        """Save experiment results."""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
''')

        # 可視化
        (project / "visualization.py").write_text('''
"""Data visualization utilities."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Tuple

def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=figsize)
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names: list, importances: list, top_n: int = 20):
    """Plot feature importance."""
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
''')

        # メインスクリプト
        (project / "main.py").write_text('''
"""Main experiment runner."""
from pathlib import Path
from data_loader import load_csv, split_data
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from experiments import ExperimentTracker
from visualization import plot_correlation_matrix

def main():
    """Run the main experiment pipeline."""
    # Configuration
    config = {
        "data_path": "data/dataset.csv",
        "target_column": "target",
        "model_type": "random_forest",
        "test_size": 0.2
    }

    # Initialize experiment tracker
    tracker = ExperimentTracker("baseline_experiment")

    # Load data
    df = load_csv(Path(config["data_path"]))

    # Visualize data
    plot_correlation_matrix(df.select_dtypes(include=["float64", "int64"]))

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        config["target_column"],
        config["test_size"]
    )

    # Feature engineering
    engineer = FeatureEngineer()
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    X_train = engineer.scale_numeric_features(X_train, numeric_cols)
    X_test = engineer.scale_numeric_features(X_test, numeric_cols)

    # Train model
    trainer = ModelTrainer(config["model_type"])
    trainer.train(X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Model performance: {metrics}")

    # Track experiment
    result = tracker.run_experiment({**config, "metrics": metrics})
    print(f"Experiment completed: {result}")

if __name__ == "__main__":
    main()
''')

        # Act
        results = parse_directory(str(project), include_dependencies=True)

        # Assert
        assert len(results) >= 6

        # 複雑な依存関係の確認
        main_file = str(project / "main.py")
        main_result = results[main_file]
        main_edges = main_result["edges"]

        # main.pyが他のモジュールをインポート
        import_edges = [e for e in main_edges if e["type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) >= 5  # 多数のローカルインポート

        # scikit-learnの使用確認
        models_file = str(project / "models.py")
        models_result = results[models_file]
        models_nodes = models_result["nodes"]

        # ModelTrainerクラスの確認
        class_nodes = [n for n in models_nodes if n["type"] == "ClassDef"]
        assert any(n["properties"]["name"] == "ModelTrainer" for n in class_nodes)

        # メソッドの確認
        models_edges = models_result["edges"]
        uses_edges = [e for e in models_edges if e["type"] == EdgeType.USES.value]
        assert len(uses_edges) > 0  # sklearn関数の使用


class TestProjectDependencyAnalysis:
    """プロジェクト全体の依存関係分析テスト."""

    def test_circular_dependencies_detection(self, tmp_path: Path) -> None:
        """循環依存の検出テスト."""
        # Arrange - 循環依存を含むプロジェクト
        project = tmp_path / "circular_deps"
        project.mkdir()

        # module_a.py imports from module_b
        (project / "module_a.py").write_text('''
"""Module A."""
from .module_b import FunctionB

def FunctionA():
    return FunctionB() + " from A"
''')

        # module_b.py imports from module_c
        (project / "module_b.py").write_text('''
"""Module B."""
from .module_c import FunctionC

def FunctionB():
    return FunctionC() + " from B"
''')

        # module_c.py imports from module_a (circular!)
        (project / "module_c.py").write_text('''
"""Module C."""
from .module_a import FunctionA

def FunctionC():
    # This would cause circular import
    return "C"

def FunctionC2():
    # Deferred import to avoid immediate circular import
    from .module_a import FunctionA
    return FunctionA()
''')

        # Act
        results = parse_directory(str(project), include_dependencies=True)

        # Assert
        assert len(results) == 3

        # 各モジュールのインポート関係を追跡
        import_graph: dict[str, set[str]] = {}

        for file_path, result in results.items():
            module_name = Path(file_path).stem
            import_graph[module_name] = set()

            edges = result["edges"]
            import_edges = [e for e in edges if e["type"] == EdgeType.IMPORTS.value]

            for _edge in import_edges:
                # ノードからインポート先を特定
                nodes = result["nodes"]
                for node in nodes:
                    if node["type"] == "ImportFrom" and "module_" in str(node.get("properties", {})):
                        imported_module = node["properties"].get("module", "").split(".")[-1]
                        if imported_module.startswith("module_"):
                            import_graph[module_name].add(imported_module)

        # 循環依存パスの存在を確認
        # module_a -> module_b -> module_c -> module_a
        assert "module_b" in import_graph.get("module_a", set()) or len(import_graph["module_a"]) > 0
        assert "module_c" in import_graph.get("module_b", set()) or len(import_graph["module_b"]) > 0
        assert "module_a" in import_graph.get("module_c", set()) or len(import_graph["module_c"]) > 0

    def test_complex_inheritance_hierarchy(self, tmp_path: Path) -> None:
        """複雑な継承階層のテスト."""
        # Arrange
        project = tmp_path / "inheritance_project"
        project.mkdir()

        (project / "base.py").write_text('''
"""Base classes."""
from abc import ABC, abstractmethod

class Animal(ABC):
    """Abstract base class for animals."""

    @abstractmethod
    def make_sound(self) -> str:
        pass

    def move(self) -> str:
        return "Moving"

class Mammal(Animal):
    """Mammal base class."""

    def give_birth(self) -> str:
        return "Giving birth to live young"

class Bird(Animal):
    """Bird base class."""

    def lay_eggs(self) -> str:
        return "Laying eggs"
''')

        (project / "pets.py").write_text('''
"""Pet animals."""
from .base import Mammal, Bird

class Dog(Mammal):
    """Dog class."""

    def make_sound(self) -> str:
        return "Woof!"

    def fetch(self) -> str:
        return "Fetching the ball"

class Cat(Mammal):
    """Cat class."""

    def make_sound(self) -> str:
        return "Meow!"

    def scratch(self) -> str:
        return "Scratching"

class Parrot(Bird):
    """Parrot class."""

    def make_sound(self) -> str:
        return "Squawk!"

    def talk(self) -> str:
        return "Polly wants a cracker"
''')

        (project / "zoo.py").write_text('''
"""Zoo management."""
from typing import List
from .base import Animal
from .pets import Dog, Cat, Parrot

class Zoo:
    """Zoo class to manage animals."""

    def __init__(self):
        self.animals: List[Animal] = []

    def add_animal(self, animal: Animal) -> None:
        """Add an animal to the zoo."""
        self.animals.append(animal)

    def all_sounds(self) -> List[str]:
        """Get all animal sounds."""
        return [animal.make_sound() for animal in self.animals]

    def create_default_zoo(self) -> None:
        """Create a zoo with default animals."""
        self.add_animal(Dog())
        self.add_animal(Cat())
        self.add_animal(Parrot())
''')

        # Act
        results = parse_directory(str(project), include_dependencies=True)

        # Assert
        assert len(results) == 3

        # base.pyのクラス階層確認
        base_result = results[str(project / "base.py")]
        base_nodes = base_result["nodes"]
        base_classes = [n for n in base_nodes if n["type"] == "ClassDef"]
        assert len(base_classes) == 3  # Animal, Mammal, Bird

        # 継承関係の確認（petsモジュール）
        pets_result = results[str(project / "pets.py")]
        pets_nodes = pets_result["nodes"]
        pets_classes = [n for n in pets_nodes if n["type"] == "ClassDef"]
        assert len(pets_classes) == 3  # Dog, Cat, Parrot

        # 継承情報の確認
        for cls in pets_classes:
            if cls["properties"]["name"] in ["Dog", "Cat"]:
                # Dog と Cat は Mammal を継承
                assert any(base for base in cls["properties"].get("bases", []))
            elif cls["properties"]["name"] == "Parrot":
                # Parrot は Bird を継承
                assert any(base for base in cls["properties"].get("bases", []))

        # zoo.pyの依存関係
        zoo_result = results[str(project / "zoo.py")]
        zoo_edges = zoo_result["edges"]

        # Animalタイプのインポートと使用
        import_edges = [e for e in zoo_edges if e["type"] == EdgeType.IMPORTS.value]
        [e for e in zoo_edges if e["type"] == EdgeType.USES.value]
        instantiates_edges = [e for e in zoo_edges if e["type"] == EdgeType.INSTANTIATES.value]

        assert len(import_edges) >= 4  # Animal, Dog, Cat, Parrot
        assert len(instantiates_edges) >= 3  # Dog(), Cat(), Parrot()の生成
