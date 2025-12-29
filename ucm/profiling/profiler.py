import json
import os
import queue
import threading
import time
import atexit
from typing import Any

from ucm.logger import init_logger

logger = init_logger(__name__)


class Profiler:
    def __init__(
        self,
        launch_config: dict[str, Any],
        block_size: int,
        rank: int,
    ) -> None:
        self.block_size = block_size
        record_config = launch_config.get("record_config", {})
        self.enable_record = record_config.get("enable", False) and rank == 0
        
        if not self.enable_record:
            return
        
        # Initialize logging components
        self.log_queue = queue.Queue(maxsize=10000)
        self.batch_buffer = []
        self.log_path = record_config.get(
            "log_path", "/home/zht/zht_3/unified-cache-management/ucm/store/test/e2e/temp/ucm_ops.log"
        )
        self.flush_size = record_config.get("flush_size", 100)
        self.flush_interval = record_config.get("flush_interval", 5.0)
        self._shutdown = threading.Event()
        
        # Speed log file path (same directory as main log, with _speed suffix)
        log_dir = os.path.dirname(self.log_path)
        log_basename = os.path.basename(self.log_path)
        if "." in log_basename:
            name, ext = log_basename.rsplit(".", 1)
            self.speed_log_path = os.path.join(log_dir, f"{name}_speed.{ext}")
        else:
            self.speed_log_path = os.path.join(log_dir, f"{log_basename}_speed")
        
        # Start background thread
        self.write_thread = threading.Thread(target=self._async_record_loop, daemon=True)
        self.write_thread.start()
        atexit.register(self._flush_on_exit)
        logger.info(f"Profiler enabled, log_path: {self.log_path}, speed_log_path: {self.speed_log_path}")

    def _flush_buffer(self):
        """Flush buffer to file"""
        if not self.batch_buffer:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for log_entry in self.batch_buffer:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self.batch_buffer.clear()
        except Exception as e:
            logger.error(f"Error writing log file: {str(e)}")

    def _flush_on_exit(self):
        """Flush remaining logs when program exits"""
        if not self.enable_record:
            return
        self._shutdown.set()
        if self.write_thread.is_alive():
            self.write_thread.join(timeout=2.0)
        self._flush_buffer()

    def log_operation(self, operation_data: dict[str, Any]) -> None:
        """Record operation log (non-blocking)"""
        if not self.enable_record:
            return

        # Prepare log entry
        log_entry = {
            "timestamp": time.time(),
            "op_type": "None",
            "block_size": self.block_size,
        }
        
        # Convert bytes to hex strings for JSON serialization
        for key, value in operation_data.items():
            if key == "blocks" and isinstance(value, list):
                log_entry[key] = [block.hex() if isinstance(block, bytes) else str(block) for block in value]
            else:
                log_entry[key] = value

        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            logger.error(f"Log queue is full, dropping log entry")

    def log_seed(self, op_type: str, value: float, count: int) -> None:
        """Record seed values (e.g., speed) - writes count lines, each with the value
        """
        if not self.enable_record:
            return
        
        if count <= 0:
            return
        
        try:
            # Directly write to speed log file (synchronous, simple format)
            # Format: one value per line
            with open(self.speed_log_path, "a", encoding="utf-8") as f:
                for _ in range(count):
                    f.write(f"{value}\n")
        except Exception as e:
            logger.error(f"Error writing speed log: {str(e)}")

    def _async_record_loop(self):
        last_flush_time = time.time()
        while not self._shutdown.is_set():
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                self.batch_buffer.append(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                pass
            
            # Check if we need to flush
            current_time = time.time()
            should_flush = (
                len(self.batch_buffer) >= self.flush_size
                or (current_time - last_flush_time) >= self.flush_interval
            )
            
            if should_flush:
                self._flush_buffer()
                last_flush_time = current_time
        
        # Flush remaining buffer on thread exit
        self._flush_buffer()