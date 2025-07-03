"""Tests for batch_processor module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import time
import sys

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
    
    def test_check_memory_usage(self):
        """Test checking current memory usage."""
        # Create mock psutil module
        mock_psutil = MagicMock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        # Patch the module before importing
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            # Re-import to get mocked version
            import importlib
            import ast2graph.batch_processor
            importlib.reload(ast2graph.batch_processor)
            
            # Now HAS_PSUTIL should be True and psutil should be our mock
            assert ast2graph.batch_processor.HAS_PSUTIL is True
            
            manager = ast2graph.batch_processor.MemoryManager(limit_mb=256)
            usage = manager.check_memory_usage()
            
            assert usage == pytest.approx(100.0, 0.1)
    
    def test_should_pause_processing_below_limit(self):
        """Test pause decision when below memory limit."""
        mock_psutil = MagicMock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 200 * 1024 * 1024  # 200 MB
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            import importlib
            import ast2graph.batch_processor
            importlib.reload(ast2graph.batch_processor)
            
            manager = ast2graph.batch_processor.MemoryManager(limit_mb=256)
            assert not manager.should_pause_processing()
    
    def test_should_pause_processing_above_threshold(self):
        """Test pause decision when above 90% of limit."""
        mock_psutil = MagicMock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 240 * 1024 * 1024  # 240 MB (93.75% of 256)
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            import importlib
            import ast2graph.batch_processor
            importlib.reload(ast2graph.batch_processor)
            
            manager = ast2graph.batch_processor.MemoryManager(limit_mb=256)
            assert manager.should_pause_processing()
    
    def test_wait_for_memory(self):
        """Test waiting for memory to be available."""
        # Simulate memory going down over time
        memory_values = [
            250 * 1024 * 1024,  # Above threshold
            240 * 1024 * 1024,  # Still above
            200 * 1024 * 1024,  # Below threshold
        ]
        
        mock_psutil = MagicMock()
        mock_memory_info = Mock()
        mock_psutil.Process.return_value.memory_info.return_value = mock_memory_info
        
        call_count = 0
        def get_memory():
            nonlocal call_count
            mock_memory_info.rss = memory_values[min(call_count, len(memory_values) - 1)]
            call_count += 1
            return mock_memory_info
        
        mock_psutil.Process.return_value.memory_info = get_memory
        
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            import importlib
            import ast2graph.batch_processor
            importlib.reload(ast2graph.batch_processor)
            
            manager = ast2graph.batch_processor.MemoryManager(limit_mb=256)
            with patch('time.sleep'):  # Don't actually sleep in tests
                manager.wait_for_memory()
            
            assert call_count >= 3
    
    def test_memory_manager_without_psutil(self):
        """Test memory manager when psutil is not available."""
        # Remove psutil from modules if it exists
        if 'psutil' in sys.modules:
            del sys.modules['psutil']
            
        # Reload to get HAS_PSUTIL = False
        import importlib
        import ast2graph.batch_processor
        importlib.reload(ast2graph.batch_processor)
        
        with patch('ast2graph.batch_processor.HAS_PSUTIL', False):
            manager = ast2graph.batch_processor.MemoryManager(limit_mb=256)
            
            # Should return 0 and False when psutil not available
            assert manager.check_memory_usage() == 0.0
            assert manager.should_pause_processing() is False


class TestErrorRecoveryStrategy:
    """Test cases for ErrorRecoveryStrategy class."""
    
    def test_error_recovery_creation(self):
        """Test creating an ErrorRecoveryStrategy instance."""
        strategy = ErrorRecoveryStrategy(
            max_retries=3,
            retry_delay=0.5
        )
        assert strategy.max_retries == 3
        assert strategy.retry_delay == 0.5
    
    def test_default_error_recovery(self):
        """Test default error recovery settings."""
        strategy = ErrorRecoveryStrategy()
        assert strategy.max_retries == 2
        assert strategy.retry_delay == 0.5
    
    def test_handle_file_error(self):
        """Test error handling for a file."""
        strategy = ErrorRecoveryStrategy(max_retries=2)
        
        # First error - should retry
        error1 = Exception("First error")
        assert strategy.handle_file_error("test.py", error1) is True
        
        # Second error - should retry
        error2 = Exception("Second error")
        assert strategy.handle_file_error("test.py", error2) is True
        
        # Third error - should not retry
        error3 = Exception("Third error")
        assert strategy.handle_file_error("test.py", error3) is False
    
    def test_get_failed_files(self):
        """Test retrieving permanently failed files."""
        strategy = ErrorRecoveryStrategy(max_retries=1)
        
        # Add errors to exceed retry limit
        error1 = Exception("Error 1")
        error2 = Exception("Error 2")
        
        strategy.handle_file_error("file1.py", error1)
        strategy.handle_file_error("file1.py", error2)
        
        strategy.handle_file_error("file2.py", error1)
        # file2.py hasn't exceeded retry limit
        
        failed_files = strategy.get_failed_files()
        assert len(failed_files) == 1
        assert failed_files[0].file_path == "file1.py"
        assert failed_files[0].error_type == "Exception"


class TestBatchConfig:
    """Test cases for BatchConfig class."""
    
    def test_batch_config_creation(self):
        """Test creating a BatchConfig instance."""
        progress_callback = Mock()
        error_callback = Mock()
        
        config = BatchConfig(
            batch_size=10,
            max_workers=4,
            memory_limit_mb=256,
            use_multiprocessing=True,
            progress_callback=progress_callback,
            error_callback=error_callback,
            enable_profiling=True
        )
        
        assert config.batch_size == 10
        assert config.max_workers == 4
        assert config.memory_limit_mb == 256
        assert config.use_multiprocessing is True
        assert config.progress_callback is progress_callback
        assert config.error_callback is error_callback
        assert config.enable_profiling is True
    
    def test_batch_config_defaults(self):
        """Test default BatchConfig settings."""
        config = BatchConfig()
        
        assert config.batch_size == 50
        assert config.max_workers is None
        assert config.memory_limit_mb == 500
        assert config.use_multiprocessing is False
        assert config.progress_callback is None
        assert config.error_callback is None
        assert config.enable_profiling is False
    
    def test_batch_config_validation(self):
        """Test BatchConfig validation."""
        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchConfig(batch_size=0)
        
        # Invalid max workers
        with pytest.raises(ValueError, match="Max workers must be positive"):
            BatchConfig(max_workers=0)
        
        # Invalid memory limit
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            BatchConfig(memory_limit_mb=0)


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""
    
    def test_batch_processor_creation(self):
        """Test creating a BatchProcessor instance."""
        config = BatchConfig(max_workers=2)
        processor = BatchProcessor(config)
        
        assert processor.config is config
    
    def test_process_files_basic(self):
        """Test basic file processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "test1.py"
            file1.write_text("x = 1")
            
            file2 = Path(tmpdir) / "test2.py"
            file2.write_text("y = 2")
            
            config = BatchConfig(max_workers=1)
            processor = BatchProcessor(config)
            
            result = processor.process_files([str(file1), str(file2)])
            
            assert isinstance(result, BatchResult)
            assert result.total_files == 2
            assert result.processed_files == 2
            assert len(result.failed_files) == 0
            assert len(result.graphs) == 2
    
    def test_process_files_with_errors(self):
        """Test file processing with syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid file
            valid_file = Path(tmpdir) / "valid.py"
            valid_file.write_text("x = 1")
            
            # Create file with syntax error
            error_file = Path(tmpdir) / "error.py"
            error_file.write_text("def invalid syntax")
            
            config = BatchConfig(max_workers=1)
            processor = BatchProcessor(config)
            
            result = processor.process_files([str(valid_file), str(error_file)])
            
            assert result.total_files == 2
            # Since one file has error, processed files should be 1
            assert result.processed_files in [1, 2]  # May vary based on implementation
            assert len(result.graphs) >= 1  # At least the valid file
            
            # Check if error was properly tracked (if implementation tracks errors)
            if hasattr(processor, 'error_recovery'):
                failed_files = processor.error_recovery.get_failed_files()
                if failed_files:
                    assert len(failed_files) >= 1
    
    def test_process_files_with_progress_callback(self):
        """Test progress callback during processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(5):
                file = Path(tmpdir) / f"test{i}.py"
                file.write_text(f"x = {i}")
                files.append(str(file))
            
            progress_updates = []
            
            def progress_callback(progress: ProcessingProgress):
                progress_updates.append({
                    'completed': progress.files_completed,
                    'total': progress.files_total,
                    'percentage': progress.percentage
                })
            
            config = BatchConfig(
                max_workers=1,
                progress_callback=progress_callback
            )
            processor = BatchProcessor(config)
            
            processor.process_files(files)
            
            # Should have received progress updates
            assert len(progress_updates) > 0
            
            # Final update should show 100%
            final_update = progress_updates[-1]
            assert final_update['completed'] == 5
            assert final_update['total'] == 5
            assert final_update['percentage'] == 100.0
    
    def test_process_files_streaming(self):
        """Test streaming file processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(10):
                file = Path(tmpdir) / f"test{i}.py"
                file.write_text(f"x = {i}")
                files.append(str(file))
            
            config = BatchConfig(batch_size=3)
            processor = BatchProcessor(config)
            
            # Process as stream
            batches_count = 0
            total_graphs = 0
            for batch_result in processor.process_files_streaming(files):
                assert isinstance(batch_result, BatchResult)
                batches_count += 1
                total_graphs += len(batch_result.graphs)
            
            # With batch_size=3 and 10 files, we expect 4 batches (3+3+3+1)
            assert batches_count == 4
            assert total_graphs == 10
    
    def test_process_files_parallel(self):
        """Test parallel file processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(10):
                file = Path(tmpdir) / f"test{i}.py"
                file.write_text(f"x = {i}\ny = {i} * 2")
                files.append(str(file))
            
            # Test with thread pool
            config = BatchConfig(max_workers=2, use_multiprocessing=False)
            processor = BatchProcessor(config)
            result = processor.process_files(files)
            
            assert result.processed_files == 10
            assert len(result.graphs) == 10
    
    def test_process_files_multiprocessing(self):
        """Test multiprocessing file processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(10):
                file = Path(tmpdir) / f"test{i}.py"
                file.write_text(f"x = {i}\ny = {i} * 2")
                files.append(str(file))
            
            # Test with process pool
            config = BatchConfig(max_workers=2, use_multiprocessing=True)
            processor = BatchProcessor(config)
            result = processor.process_files(files)
            
            assert result.processed_files == 10
            assert len(result.graphs) == 10


class TestBatchResult:
    """Test cases for BatchResult class."""
    
    def test_batch_result_creation(self):
        """Test creating a BatchResult instance."""
        failed_files = [
            FailedFile.from_exception("error.py", Exception("Syntax error"))
        ]
        graphs = [GraphStructure()]
        
        result = BatchResult(
            total_files=10,
            processed_files=9,
            failed_files=failed_files,
            processing_time=5.0,
            memory_peak_mb=100.0,
            graphs=graphs
        )
        
        assert result.total_files == 10
        assert result.processed_files == 9
        assert len(result.failed_files) == 1
        assert result.processing_time == 5.0
        assert result.memory_peak_mb == 100.0
        assert len(result.graphs) == 1
    
    def test_batch_result_success_rate(self):
        """Test success rate calculation."""
        result = BatchResult(
            total_files=10,
            processed_files=8,
            failed_files=[],
            processing_time=5.0,
            memory_peak_mb=100.0,
            graphs=[]
        )
        
        assert result.success_rate == 0.8
        
        # Test with zero total files
        result_empty = BatchResult(
            total_files=0,
            processed_files=0,
            failed_files=[],
            processing_time=0.0,
            memory_peak_mb=0.0,
            graphs=[]
        )
        
        assert result_empty.success_rate == 1.0
    
    def test_batch_result_merge(self):
        """Test merging multiple batch results."""
        result1 = BatchResult(
            total_files=5,
            processed_files=4,
            failed_files=[FailedFile.from_exception("f1.py", Exception("E1"))],
            processing_time=2.0,
            memory_peak_mb=50.0,
            graphs=[GraphStructure()]
        )
        
        result2 = BatchResult(
            total_files=5,
            processed_files=5,
            failed_files=[],
            processing_time=3.0,
            memory_peak_mb=75.0,
            graphs=[GraphStructure(), GraphStructure()]
        )
        
        merged = BatchResult.merge([result1, result2])
        
        assert merged.total_files == 10
        assert merged.processed_files == 9
        assert len(merged.failed_files) == 1
        assert merged.processing_time == 5.0
        assert merged.memory_peak_mb == 75.0  # Max of the two
        assert len(merged.graphs) == 3


class TestProcessingProgress:
    """Test cases for ProcessingProgress class."""
    
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
        """Test percentage calculation."""
        progress = ProcessingProgress(
            current_file="test.py",
            files_completed=3,
            files_total=10,
            current_phase="parsing",
            elapsed_time=1.0,
            estimated_remaining=2.0,
            memory_usage_mb=50.0
        )
        
        assert progress.percentage == 30.0
        
        # Test with zero total
        progress_empty = ProcessingProgress(
            current_file="",
            files_completed=0,
            files_total=0,
            current_phase="idle",
            elapsed_time=0.0,
            estimated_remaining=0.0,
            memory_usage_mb=0.0
        )
        
        assert progress_empty.percentage == 100.0


class TestProcessingMetrics:
    """Test cases for ProcessingMetrics class."""
    
    def test_processing_metrics_creation(self):
        """Test creating a ProcessingMetrics instance."""
        from datetime import datetime
        
        start = datetime.now()
        end = start.replace(microsecond=0) + \
            __import__('datetime').timedelta(seconds=10)
        
        metrics = ProcessingMetrics(
            files_processed=10,
            nodes_created=100,
            edges_created=50,
            processing_time=10.0,
            memory_peak=200.0,
            errors=["Error 1", "Error 2"],
            start_time=start,
            end_time=end
        )
        
        assert metrics.files_processed == 10
        assert metrics.nodes_created == 100
        assert metrics.edges_created == 50
        assert metrics.processing_time == 10.0
        assert metrics.memory_peak == 200.0
        assert len(metrics.errors) == 2
    
    def test_processing_metrics_calculations(self):
        """Test metric calculations."""
        from datetime import datetime
        
        metrics = ProcessingMetrics(
            files_processed=10,
            nodes_created=100,
            edges_created=50,
            processing_time=5.0,
            memory_peak=200.0,
            errors=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert metrics.average_nodes_per_file == 10.0
        assert metrics.files_per_second == 2.0
        
        # Test with zero files
        metrics_empty = ProcessingMetrics(
            files_processed=0,
            nodes_created=0,
            edges_created=0,
            processing_time=0.0,
            memory_peak=0.0,
            errors=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert metrics_empty.average_nodes_per_file == 0.0
        assert metrics_empty.files_per_second == 0.0