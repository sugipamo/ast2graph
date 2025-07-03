"""Tests for batch_types module."""
from datetime import datetime

import pytest

from ast2graph.batch_types import (
    BatchConfig,
    BatchResult,
    FailedFile,
    ProcessingMetrics,
    ProcessingProgress,
)
from ast2graph.graph_structure import GraphStructure


class TestFailedFile:
    """Test cases for FailedFile dataclass."""

    def test_failed_file_creation(self):
        """Test creating a FailedFile instance."""
        error = ValueError("Invalid syntax")
        failed = FailedFile(
            file_path="/path/to/file.py",
            error=error,
            error_type="ValueError",
            error_message="Invalid syntax"
        )

        assert failed.file_path == "/path/to/file.py"
        assert failed.error == error
        assert failed.error_type == "ValueError"
        assert failed.error_message == "Invalid syntax"

    def test_failed_file_from_exception(self):
        """Test creating FailedFile from exception."""
        try:
            raise RuntimeError("Test error")
        except Exception as e:
            failed = FailedFile.from_exception("/test/file.py", e)

            assert failed.file_path == "/test/file.py"
            assert isinstance(failed.error, RuntimeError)
            assert failed.error_type == "RuntimeError"
            assert failed.error_message == "Test error"


class TestBatchResult:
    """Test cases for BatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test creating a BatchResult instance."""
        graph1 = GraphStructure()
        graph2 = GraphStructure()
        failed = FailedFile.from_exception("file3.py", ValueError("Error"))

        result = BatchResult(
            total_files=3,
            processed_files=2,
            failed_files=[failed],
            processing_time=1.5,
            memory_peak_mb=100.0,
            graphs=[graph1, graph2]
        )

        assert result.total_files == 3
        assert result.processed_files == 2
        assert len(result.failed_files) == 1
        assert result.processing_time == 1.5
        assert result.memory_peak_mb == 100.0
        assert len(result.graphs) == 2

    def test_batch_result_success_rate(self):
        """Test calculating success rate."""
        result = BatchResult(
            total_files=10,
            processed_files=7,
            failed_files=[],
            processing_time=5.0,
            memory_peak_mb=200.0,
            graphs=[]
        )

        assert result.success_rate == 0.7

    def test_batch_result_success_rate_zero_files(self):
        """Test success rate with zero files."""
        result = BatchResult(
            total_files=0,
            processed_files=0,
            failed_files=[],
            processing_time=0.0,
            memory_peak_mb=0.0,
            graphs=[]
        )

        assert result.success_rate == 1.0  # No files = 100% success

    def test_batch_result_merge(self):
        """Test merging multiple BatchResults."""
        result1 = BatchResult(
            total_files=5,
            processed_files=4,
            failed_files=[FailedFile.from_exception("f1.py", ValueError())],
            processing_time=2.0,
            memory_peak_mb=100.0,
            graphs=[GraphStructure(), GraphStructure()]
        )

        result2 = BatchResult(
            total_files=3,
            processed_files=2,
            failed_files=[FailedFile.from_exception("f2.py", RuntimeError())],
            processing_time=1.5,
            memory_peak_mb=150.0,
            graphs=[GraphStructure()]
        )

        merged = BatchResult.merge([result1, result2])

        assert merged.total_files == 8
        assert merged.processed_files == 6
        assert len(merged.failed_files) == 2
        assert merged.processing_time == 3.5
        assert merged.memory_peak_mb == 150.0  # Max of both
        assert len(merged.graphs) == 3


class TestProcessingProgress:
    """Test cases for ProcessingProgress dataclass."""

    def test_processing_progress_creation(self):
        """Test creating a ProcessingProgress instance."""
        progress = ProcessingProgress(
            current_file="test.py",
            files_completed=5,
            files_total=10,
            current_phase="parsing",
            elapsed_time=2.5,
            estimated_remaining=2.5,
            memory_usage_mb=100.0
        )

        assert progress.current_file == "test.py"
        assert progress.files_completed == 5
        assert progress.files_total == 10
        assert progress.current_phase == "parsing"
        assert progress.elapsed_time == 2.5
        assert progress.estimated_remaining == 2.5
        assert progress.memory_usage_mb == 100.0

    def test_processing_progress_percentage(self):
        """Test calculating progress percentage."""
        progress = ProcessingProgress(
            current_file="test.py",
            files_completed=3,
            files_total=10,
            current_phase="parsing",
            elapsed_time=1.0,
            estimated_remaining=2.33,
            memory_usage_mb=50.0
        )

        assert progress.percentage == 30.0

    def test_processing_progress_percentage_zero_total(self):
        """Test progress percentage with zero total files."""
        progress = ProcessingProgress(
            current_file="",
            files_completed=0,
            files_total=0,
            current_phase="idle",
            elapsed_time=0.0,
            estimated_remaining=0.0,
            memory_usage_mb=0.0
        )

        assert progress.percentage == 100.0  # No work = 100% done


class TestProcessingMetrics:
    """Test cases for ProcessingMetrics dataclass."""

    def test_processing_metrics_creation(self):
        """Test creating a ProcessingMetrics instance."""
        start_time = datetime.now()
        metrics = ProcessingMetrics(
            files_processed=10,
            nodes_created=500,
            edges_created=800,
            processing_time=5.0,
            memory_peak=256.0,
            errors=[],
            start_time=start_time,
            end_time=datetime.now()
        )

        assert metrics.files_processed == 10
        assert metrics.nodes_created == 500
        assert metrics.edges_created == 800
        assert metrics.processing_time == 5.0
        assert metrics.memory_peak == 256.0
        assert len(metrics.errors) == 0
        assert metrics.start_time == start_time

    def test_processing_metrics_average_nodes_per_file(self):
        """Test calculating average nodes per file."""
        metrics = ProcessingMetrics(
            files_processed=10,
            nodes_created=500,
            edges_created=800,
            processing_time=5.0,
            memory_peak=256.0,
            errors=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        assert metrics.average_nodes_per_file == 50.0

    def test_processing_metrics_average_nodes_zero_files(self):
        """Test average nodes with zero files processed."""
        metrics = ProcessingMetrics(
            files_processed=0,
            nodes_created=0,
            edges_created=0,
            processing_time=0.0,
            memory_peak=0.0,
            errors=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        assert metrics.average_nodes_per_file == 0.0

    def test_processing_metrics_files_per_second(self):
        """Test calculating files per second."""
        metrics = ProcessingMetrics(
            files_processed=20,
            nodes_created=1000,
            edges_created=1500,
            processing_time=10.0,
            memory_peak=512.0,
            errors=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        assert metrics.files_per_second == 2.0


class TestBatchConfig:
    """Test cases for BatchConfig dataclass."""

    def test_batch_config_defaults(self):
        """Test BatchConfig with default values."""
        config = BatchConfig()

        assert config.batch_size == 50
        assert config.max_workers is None
        assert config.memory_limit_mb == 500
        assert config.use_multiprocessing is False
        assert config.progress_callback is None
        assert config.error_callback is None
        assert config.enable_profiling is False

    def test_batch_config_custom_values(self):
        """Test BatchConfig with custom values."""
        def progress_cb(progress): pass
        def error_cb(error): pass

        config = BatchConfig(
            batch_size=100,
            max_workers=8,
            memory_limit_mb=1024,
            use_multiprocessing=True,
            progress_callback=progress_cb,
            error_callback=error_cb,
            enable_profiling=True
        )

        assert config.batch_size == 100
        assert config.max_workers == 8
        assert config.memory_limit_mb == 1024
        assert config.use_multiprocessing is True
        assert config.progress_callback == progress_cb
        assert config.error_callback == error_cb
        assert config.enable_profiling is True

    def test_batch_config_validate_batch_size(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchConfig(batch_size=0)

        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchConfig(batch_size=-1)

    def test_batch_config_validate_max_workers(self):
        """Test max workers validation."""
        with pytest.raises(ValueError, match="Max workers must be positive"):
            BatchConfig(max_workers=0)

        with pytest.raises(ValueError, match="Max workers must be positive"):
            BatchConfig(max_workers=-1)

    def test_batch_config_validate_memory_limit(self):
        """Test memory limit validation."""
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            BatchConfig(memory_limit_mb=0)

        with pytest.raises(ValueError, match="Memory limit must be positive"):
            BatchConfig(memory_limit_mb=-100)
