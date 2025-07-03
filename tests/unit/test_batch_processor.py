"""Tests for batch_processor module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import time

from ast2graph.batch_processor import BatchProcessor, MemoryManager, ErrorRecoveryStrategy
from ast2graph.batch_types import (
    BatchConfig, BatchResult, FailedFile, ProcessingProgress,
    ProcessingMetrics
)
from ast2graph.graph_structure import GraphStructure
from ast2graph.exceptions import BatchProcessingError


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    def test_memory_manager_creation(self):
        """Test creating a MemoryManager instance."""
        manager = MemoryManager(limit_mb=256)
        assert manager.limit_mb == 256
    
    @patch('ast2graph.batch_processor.HAS_PSUTIL', True)
    @patch('ast2graph.batch_processor.psutil')
    def test_check_memory_usage(self, mock_psutil):
        """Test checking current memory usage."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        manager = MemoryManager(limit_mb=256)
        usage = manager.check_memory_usage()
        
        assert usage == pytest.approx(100.0, 0.1)
    
    @patch('ast2graph.batch_processor.HAS_PSUTIL', True)
    @patch('ast2graph.batch_processor.psutil')
    def test_should_pause_processing_below_limit(self, mock_psutil):
        """Test pause decision when below memory limit."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 200 * 1024 * 1024  # 200 MB
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        manager = MemoryManager(limit_mb=256)
        assert not manager.should_pause_processing()
    
    @patch('ast2graph.batch_processor.HAS_PSUTIL', True)
    @patch('ast2graph.batch_processor.psutil')
    def test_should_pause_processing_above_threshold(self, mock_psutil):
        """Test pause decision when above 90% of limit."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 240 * 1024 * 1024  # 240 MB (93.75% of 256)
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        manager = MemoryManager(limit_mb=256)
        assert manager.should_pause_processing()
    
    @patch('ast2graph.batch_processor.HAS_PSUTIL', True)
    @patch('ast2graph.batch_processor.psutil')
    def test_wait_for_memory(self, mock_psutil):
        """Test waiting for memory to be available."""
        # Simulate memory going down over time
        memory_values = [
            250 * 1024 * 1024,  # Above threshold
            240 * 1024 * 1024,  # Still above
            200 * 1024 * 1024,  # Below threshold
        ]
        mock_memory_info = Mock()
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        call_count = 0
        def get_memory():
            nonlocal call_count
            mock_memory_info.rss = memory_values[min(call_count, len(memory_values) - 1)]
            call_count += 1
            return mock_memory_info
        
        mock_psutil.Process.return_value.memory_info = get_memory
        
        manager = MemoryManager(limit_mb=256)
        with patch('time.sleep'):  # Don't actually sleep in tests
            manager.wait_for_memory()
        
        assert call_count >= 3
    
    def test_memory_manager_without_psutil(self):
        """Test memory manager when psutil is not available."""
        with patch('ast2graph.batch_processor.HAS_PSUTIL', False):
            manager = MemoryManager(limit_mb=256)
            
            # Should return 0 and False when psutil not available
            assert manager.check_memory_usage() == 0.0
            assert manager.should_pause_processing() is False


class TestErrorRecoveryStrategy:
    """Test cases for ErrorRecoveryStrategy class."""
    
    def test_error_recovery_creation(self):
        """Test creating an ErrorRecoveryStrategy instance."""
        strategy = ErrorRecoveryStrategy(max_retries=3, retry_delay=1.0)
        assert strategy.max_retries == 3
        assert strategy.retry_delay == 1.0
        assert len(strategy.failed_files) == 0
    
    def test_handle_file_error_first_attempt(self):
        """Test handling error on first attempt."""
        strategy = ErrorRecoveryStrategy(max_retries=3)
        
        should_retry = strategy.handle_file_error(
            "test.py",
            ValueError("Syntax error")
        )
        
        assert should_retry is True
        assert "test.py" in strategy.failed_files
        assert strategy.failed_files["test.py"]["attempts"] == 1
    
    def test_handle_file_error_max_retries(self):
        """Test handling error after max retries."""
        strategy = ErrorRecoveryStrategy(max_retries=2)
        
        # First attempt
        assert strategy.handle_file_error("test.py", ValueError()) is True
        # Second attempt
        assert strategy.handle_file_error("test.py", ValueError()) is True
        # Third attempt - should not retry
        assert strategy.handle_file_error("test.py", ValueError()) is False
        
        assert strategy.failed_files["test.py"]["attempts"] == 3
    
    def test_get_failed_files(self):
        """Test getting list of permanently failed files."""
        strategy = ErrorRecoveryStrategy(max_retries=1)
        
        # File 1: will exceed retry limit
        strategy.handle_file_error("file1.py", ValueError("Error 1"))
        strategy.handle_file_error("file1.py", ValueError("Error 2"))
        
        # File 2: still within retry limit
        strategy.handle_file_error("file2.py", RuntimeError("Error"))
        
        failed = strategy.get_failed_files()
        assert len(failed) == 1
        assert failed[0].file_path == "file1.py"


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""
    
    def test_batch_processor_creation_defaults(self):
        """Test creating BatchProcessor with default config."""
        processor = BatchProcessor()
        assert processor.config.batch_size == 50
        assert processor.config.memory_limit_mb == 500
    
    def test_batch_processor_creation_custom_config(self):
        """Test creating BatchProcessor with custom config."""
        config = BatchConfig(
            batch_size=100,
            max_workers=4,
            memory_limit_mb=1024
        )
        processor = BatchProcessor(config)
        assert processor.config.batch_size == 100
        assert processor.config.max_workers == 4
    
    def test_split_into_batches(self):
        """Test splitting files into batches."""
        processor = BatchProcessor(BatchConfig(batch_size=3))
        files = ["f1.py", "f2.py", "f3.py", "f4.py", "f5.py"]
        
        batches = processor._split_into_batches(files)
        
        assert len(batches) == 2
        assert batches[0] == ["f1.py", "f2.py", "f3.py"]
        assert batches[1] == ["f4.py", "f5.py"]
    
    def test_split_into_batches_empty(self):
        """Test splitting empty file list."""
        processor = BatchProcessor()
        batches = processor._split_into_batches([])
        assert batches == []
    
    @patch('ast2graph.batch_processor.parse_file')
    def test_process_single_file_success(self, mock_parse):
        """Test processing a single file successfully."""
        mock_graph = GraphStructure()
        mock_parse.return_value = mock_graph
        
        processor = BatchProcessor()
        result = processor._process_single_file("test.py")
        
        assert result == mock_graph
        mock_parse.assert_called_once_with("test.py", output_format="graph")
    
    @patch('ast2graph.batch_processor.parse_file')
    def test_process_single_file_error(self, mock_parse):
        """Test processing a single file with error."""
        mock_parse.side_effect = ValueError("Syntax error")
        
        processor = BatchProcessor()
        result = processor._process_single_file("test.py")
        
        assert result is None
        assert "test.py" in processor.error_recovery.failed_files
    
    @patch('ast2graph.batch_processor.BatchProcessor._process_single_file')
    def test_process_batch_sequential(self, mock_process):
        """Test processing a batch sequentially."""
        mock_process.side_effect = [
            GraphStructure(),  # file1 success
            None,              # file2 failure
            GraphStructure(),  # file3 success
        ]
        
        processor = BatchProcessor(BatchConfig(use_multiprocessing=False))
        files = ["file1.py", "file2.py", "file3.py"]
        
        graphs = processor._process_batch(files)
        
        assert len(graphs) == 2  # Only successful ones
        assert mock_process.call_count == 3
    
    def test_process_files_empty_list(self):
        """Test processing empty file list."""
        processor = BatchProcessor()
        result = processor.process_files([])
        
        assert result.total_files == 0
        assert result.processed_files == 0
        assert len(result.failed_files) == 0
        assert len(result.graphs) == 0
    
    @patch('ast2graph.batch_processor.parse_file')
    def test_process_files_with_progress_callback(self, mock_parse):
        """Test progress callback is called during processing."""
        mock_parse.return_value = GraphStructure()
        
        progress_updates = []
        def progress_callback(progress: ProcessingProgress):
            progress_updates.append(progress)
        
        config = BatchConfig(
            batch_size=2,
            progress_callback=progress_callback
        )
        processor = BatchProcessor(config)
        
        files = ["f1.py", "f2.py", "f3.py"]
        result = processor.process_files(files)
        
        assert len(progress_updates) > 0
        # Check final progress
        final_progress = progress_updates[-1]
        assert final_progress.files_completed == 3
        assert final_progress.files_total == 3
    
    @patch('ast2graph.batch_processor.parse_file')
    def test_process_files_with_error_callback(self, mock_parse):
        """Test error callback is called for failures."""
        # Create a function that returns the expected values
        def parse_side_effect(file_path, output_format="dict"):
            if file_path == "f2.py":
                raise ValueError("Error in file2")
            return GraphStructure()
        
        mock_parse.side_effect = parse_side_effect
        
        error_callbacks = []
        def error_callback(failed: FailedFile):
            error_callbacks.append(failed)
        
        config = BatchConfig(
            error_callback=error_callback,
            use_multiprocessing=False
        )
        processor = BatchProcessor(config)
        
        files = ["f1.py", "f2.py", "f3.py"]
        result = processor.process_files(files)
        
        # Error callback may be called multiple times due to retries
        assert len(error_callbacks) >= 1
        assert any(ec.file_path == "f2.py" for ec in error_callbacks)
        assert any("Error in file2" in ec.error_message for ec in error_callbacks)
    
    @patch('ast2graph.batch_processor.parse_file')
    @patch('ast2graph.batch_processor.HAS_PSUTIL', True)
    @patch('ast2graph.batch_processor.psutil')
    def test_memory_management_during_processing(self, mock_psutil, mock_parse):
        """Test memory management during batch processing."""
        # Simulate high memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 450 * 1024 * 1024  # 450 MB (90% of 500)
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        mock_parse.return_value = GraphStructure()
        
        config = BatchConfig(
            batch_size=10,
            memory_limit_mb=500,
            use_multiprocessing=False
        )
        processor = BatchProcessor(config)
        
        # Should process but with memory checks
        files = ["f1.py", "f2.py"]
        result = processor.process_files(files)
        
        assert result.processed_files == 2
        # Memory manager should have been consulted
        assert mock_psutil.Process.called
    
    def test_process_files_with_profiling(self):
        """Test processing with profiling enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(3):
                filepath = os.path.join(tmpdir, f"test{i}.py")
                with open(filepath, 'w') as f:
                    f.write(f"x = {i}")
                files.append(filepath)
            
            config = BatchConfig(
                enable_profiling=True,
                use_multiprocessing=False
            )
            processor = BatchProcessor(config)
            
            result = processor.process_files(files)
            
            assert result.total_files == 3
            assert result.processed_files == 3
            # Check that metrics were collected
            assert hasattr(processor, 'last_metrics')
            assert processor.last_metrics.files_processed == 3
    
    @patch('ast2graph.batch_processor.parse_file')
    def test_process_files_streaming(self, mock_parse):
        """Test streaming processing of files."""
        mock_parse.return_value = GraphStructure()
        
        processor = BatchProcessor(BatchConfig(batch_size=2))
        files = ["f1.py", "f2.py", "f3.py", "f4.py", "f5.py"]
        
        results = list(processor.process_files_streaming(files))
        
        assert len(results) == 3  # 3 batches: [2, 2, 1]
        total_processed = sum(r.processed_files for r in results)
        assert total_processed == 5
    
    def test_get_metrics(self):
        """Test getting processing metrics."""
        processor = BatchProcessor()
        
        # Before any processing
        metrics = processor.get_metrics()
        assert metrics is None
        
        # After processing (mocked)
        processor.last_metrics = ProcessingMetrics(
            files_processed=10,
            nodes_created=500,
            edges_created=750,
            processing_time=5.0,
            memory_peak=256.0,
            errors=[],
            start_time=None,
            end_time=None
        )
        
        metrics = processor.get_metrics()
        assert metrics.files_processed == 10
        assert metrics.nodes_created == 500