"""Batch processing functionality for ast2graph."""
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .api import parse_file
from .batch_types import BatchConfig, BatchResult, FailedFile, ProcessingMetrics, ProcessingProgress
from .graph_structure import GraphStructure


class MemoryManager:
    """Manages memory usage during batch processing."""

    def __init__(self, limit_mb: int = 500):
        """Initialize memory manager.

        Args:
            limit_mb: Memory limit in megabytes
        """
        self.limit_mb = limit_mb
        if not HAS_PSUTIL:
            import warnings
            warnings.warn(
                "psutil not available, memory management will be disabled. "
                "Install with: pip install psutil",
                RuntimeWarning, stacklevel=2
            )

    def check_memory_usage(self) -> float:
        """Check current memory usage in MB.

        Returns:
            Current memory usage in megabytes
        """
        if not HAS_PSUTIL:
            return 0.0

        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB

    def should_pause_processing(self) -> bool:
        """Check if processing should pause due to memory constraints.

        Returns:
            True if memory usage is above 90% of limit
        """
        if not HAS_PSUTIL:
            return False

        current_usage = self.check_memory_usage()
        threshold = self.limit_mb * 0.9  # 90% threshold
        return current_usage > threshold

    def wait_for_memory(self, check_interval: float = 1.0):
        """Wait until memory usage is below threshold.

        Args:
            check_interval: Seconds between memory checks
        """
        while self.should_pause_processing():
            time.sleep(check_interval)


class ErrorRecoveryStrategy:
    """Handles error recovery and retry logic."""

    def __init__(self, max_retries: int = 2, retry_delay: float = 0.5):
        """Initialize error recovery strategy.

        Args:
            max_retries: Maximum number of retries per file
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.failed_files: dict[str, dict[str, Any]] = {}

    def handle_file_error(self, file_path: str, error: Exception) -> bool:
        """Handle error for a file and decide whether to retry.

        Args:
            file_path: Path to the failed file
            error: The exception that occurred

        Returns:
            True if the file should be retried, False otherwise
        """
        if file_path not in self.failed_files:
            self.failed_files[file_path] = {
                "attempts": 0,
                "errors": []
            }

        file_info = self.failed_files[file_path]
        file_info["attempts"] += 1
        file_info["errors"].append({
            "error": error,
            "timestamp": datetime.now()
        })

        return file_info["attempts"] <= self.max_retries

    def get_failed_files(self) -> list[FailedFile]:
        """Get list of files that permanently failed.

        Returns:
            List of FailedFile objects for files that exceeded retry limit
        """
        failed = []
        for file_path, info in self.failed_files.items():
            if info["attempts"] > self.max_retries:
                last_error = info["errors"][-1]["error"]
                failed.append(FailedFile.from_exception(file_path, last_error))
        return failed


class BatchProcessor:
    """Processes multiple files in batches with parallel execution."""

    def __init__(self, config: BatchConfig | None = None):
        """Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.memory_manager = MemoryManager(self.config.memory_limit_mb)
        self.error_recovery = ErrorRecoveryStrategy()
        self.last_metrics: ProcessingMetrics | None = None

    def _split_into_batches(self, file_paths: list[str]) -> list[list[str]]:
        """Split file paths into batches.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of batches, each containing file paths
        """
        if not file_paths:
            return []

        batches = []
        for i in range(0, len(file_paths), self.config.batch_size):
            batch = file_paths[i:i + self.config.batch_size]
            batches.append(batch)

        return batches

    def _process_single_file(self, file_path: str) -> GraphStructure | None:
        """Process a single file.

        Args:
            file_path: Path to the file to process

        Returns:
            GraphStructure if successful, None if failed
        """
        try:
            return parse_file(file_path, output_format="graph")
        except Exception as e:
            should_retry = self.error_recovery.handle_file_error(file_path, e)

            if self.config.error_callback:
                failed_file = FailedFile.from_exception(file_path, e)
                self.config.error_callback(failed_file)

            if should_retry:
                time.sleep(self.error_recovery.retry_delay)
                try:
                    return parse_file(file_path, output_format="graph")
                except Exception as retry_error:
                    self.error_recovery.handle_file_error(file_path, retry_error)

            return None

    def _process_batch(self, file_paths: list[str]) -> list[GraphStructure]:
        """Process a batch of files.

        Args:
            file_paths: List of file paths in the batch

        Returns:
            List of successfully processed GraphStructures
        """
        graphs = []

        if self.config.use_multiprocessing:
            max_workers = self.config.max_workers or cpu_count()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(parse_file, fp, output_format="graph"): fp
                    for fp in file_paths
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        graph = future.result()
                        if graph:
                            graphs.append(graph)
                    except Exception as e:
                        self.error_recovery.handle_file_error(file_path, e)
                        if self.config.error_callback:
                            self.config.error_callback(
                                FailedFile.from_exception(file_path, e)
                            )
        else:
            # Sequential processing
            for file_path in file_paths:
                self.memory_manager.wait_for_memory()
                graph = self._process_single_file(file_path)
                if graph:
                    graphs.append(graph)

        return graphs

    def process_files(self, file_paths: list[str]) -> BatchResult:
        """Process multiple files in batches.

        Args:
            file_paths: List of file paths to process

        Returns:
            BatchResult containing processing results
        """
        start_time = time.time()
        batches = self._split_into_batches(file_paths)
        total_files = len(file_paths)
        all_graphs = []
        files_processed = 0

        # Track metrics
        nodes_created = 0
        edges_created = 0
        errors = []
        memory_peak = 0.0

        for _batch_idx, batch in enumerate(batches):
            # Report progress
            if self.config.progress_callback:
                progress = ProcessingProgress(
                    current_file=batch[0] if batch else "",
                    files_completed=files_processed,
                    files_total=total_files,
                    current_phase="processing",
                    elapsed_time=time.time() - start_time,
                    estimated_remaining=self._estimate_remaining_time(
                        files_processed, total_files, time.time() - start_time
                    ),
                    memory_usage_mb=self.memory_manager.check_memory_usage()
                )
                self.config.progress_callback(progress)

            # Process batch
            batch_graphs = self._process_batch(batch)
            all_graphs.extend(batch_graphs)
            files_processed += len(batch)

            # Update metrics
            for graph in batch_graphs:
                nodes_created += len(graph.nodes)
                edges_created += len(graph.edges)

            current_memory = self.memory_manager.check_memory_usage()
            memory_peak = max(memory_peak, current_memory)

        # Final progress report
        if self.config.progress_callback:
            progress = ProcessingProgress(
                current_file="",
                files_completed=files_processed,
                files_total=total_files,
                current_phase="completed",
                elapsed_time=time.time() - start_time,
                estimated_remaining=0.0,
                memory_usage_mb=self.memory_manager.check_memory_usage()
            )
            self.config.progress_callback(progress)

        processing_time = time.time() - start_time
        failed_files = self.error_recovery.get_failed_files()
        processed_files = len(all_graphs)

        # Store metrics if profiling enabled
        if self.config.enable_profiling:
            self.last_metrics = ProcessingMetrics(
                files_processed=processed_files,
                nodes_created=nodes_created,
                edges_created=edges_created,
                processing_time=processing_time,
                memory_peak=memory_peak,
                errors=[str(e) for e in errors],
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now()
            )

        return BatchResult(
            total_files=total_files,
            processed_files=processed_files,
            failed_files=failed_files,
            processing_time=processing_time,
            memory_peak_mb=memory_peak,
            graphs=all_graphs
        )

    def process_files_streaming(
        self,
        file_paths: list[str],
        chunk_size: int | None = None
    ) -> Iterator[BatchResult]:
        """Process files in streaming mode, yielding results for each batch.

        Args:
            file_paths: List of file paths to process
            chunk_size: Size of each chunk (defaults to batch_size)

        Yields:
            BatchResult for each processed batch
        """
        chunk_size = chunk_size or self.config.batch_size
        batches = self._split_into_batches(file_paths)

        for batch in batches:
            batch_result = self.process_files(batch)
            yield batch_result

    def get_metrics(self) -> ProcessingMetrics | None:
        """Get metrics from the last processing run.

        Returns:
            ProcessingMetrics if profiling was enabled, None otherwise
        """
        return self.last_metrics

    def _estimate_remaining_time(
        self,
        files_completed: int,
        files_total: int,
        elapsed_time: float
    ) -> float:
        """Estimate remaining processing time.

        Args:
            files_completed: Number of files completed
            files_total: Total number of files
            elapsed_time: Time elapsed so far

        Returns:
            Estimated remaining time in seconds
        """
        if files_completed == 0:
            return 0.0

        files_remaining = files_total - files_completed
        rate = files_completed / elapsed_time  # files per second

        if rate == 0:
            return 0.0

        return files_remaining / rate
