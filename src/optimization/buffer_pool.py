"""
High-Performance Frame Buffer Pool and Zero-Allocation Array Recycler.
Eliminates Garbage Collector latency spikes and memory fragmentation during
continuous 30+ FPS video and camera pipeline execution.
"""

import threading
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("BufferPool")


class FrameBufferPool:
    """
    Thread-safe memory buffer pool for ndarray reuse.
    Maintains buckets of pre-allocated numpy arrays indexed by (shape, dtype).
    """

    _instance: Optional["FrameBufferPool"] = None
    _lock = threading.Lock()

    def __init__(self, max_buffers_per_spec: int = 16):
        """
        Args:
            max_buffers_per_spec: Maximum idle buffers retained per (shape, dtype) bucket.
        """
        self.max_buffers_per_spec = max_buffers_per_spec
        self._pools: Dict[Tuple[Tuple[int, ...], np.dtype], List[np.ndarray]] = {}
        self._pool_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def get_instance(cls) -> "FrameBufferPool":
        """Singleton accessor for global buffer pool."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def acquire(
        self,
        shape: Tuple[int, ...],
        dtype: Any = np.uint8,
        zero_fill: bool = False,
    ) -> np.ndarray:
        """
        Acquires a buffer matching the requested shape and dtype from pool,
        or creates a new one if none are available.
        """
        np_dtype = np.dtype(dtype)
        key = (tuple(shape), np_dtype)

        buf = None
        with self._pool_lock:
            bucket = self._pools.get(key)
            if bucket and len(bucket) > 0:
                buf = bucket.pop()
                self._hits += 1
            else:
                self._misses += 1

        if buf is None:
            buf = np.empty(shape, dtype=np_dtype)

        if zero_fill:
            buf.fill(0)

        return buf

    def release(self, buf: np.ndarray) -> None:
        """Returns a buffer to the pool for reuse."""
        if buf is None or not isinstance(buf, np.ndarray):
            return

        key = (tuple(buf.shape), buf.dtype)
        with self._pool_lock:
            bucket = self._pools.setdefault(key, [])
            if len(bucket) < self.max_buffers_per_spec:
                bucket.append(buf)

    @contextmanager
    def scoped_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: Any = np.uint8,
        zero_fill: bool = False,
    ):
        """Context manager that automatically releases buffer back to pool upon exit."""
        buf = self.acquire(shape, dtype=dtype, zero_fill=zero_fill)
        try:
            yield buf
        finally:
            self.release(buf)

    def clear(self) -> None:
        """Frees all idle pooled buffers."""
        with self._pool_lock:
            self._pools.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns allocation telemetry and hit ratio."""
        with self._pool_lock:
            total_idle = sum(len(b) for b in self._pools.values())
            total_requests = self._hits + self._misses
            hit_ratio = (self._hits / total_requests) if total_requests > 0 else 0.0
            return {
                "total_idle_buffers": total_idle,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio_percent": round(hit_ratio * 100.0, 1),
                "buckets": {f"{k[0]}_{k[1].name}": len(v) for k, v in self._pools.items()},
            }


_global_pool: Optional[FrameBufferPool] = None


def get_buffer_pool() -> FrameBufferPool:
    """Helper to access global buffer pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = FrameBufferPool.get_instance()
    return _global_pool
