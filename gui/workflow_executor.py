"""Workflow graph execution: topological run with per-node handlers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal


NodeHandler = Callable[[str, "WorkflowExecutor"], None]
# Handler signature: (node_key, executor) -> None
# Must eventually call executor.finish_node(key, ok, message)


def topological_order(node_keys: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn topo-sort. On cycle, fall back to input order for remaining nodes."""
    keys = list(dict.fromkeys(node_keys))
    keyset = set(keys)
    indeg = {k: 0 for k in keys}
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        if a in keyset and b in keyset and a != b:
            adj[a].append(b)
            indeg[b] += 1

    q = deque([k for k in keys if indeg[k] == 0])
    order: list[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) < len(keys):
        # Cycle — append leftovers in original order
        leftover = [k for k in keys if k not in order]
        order.extend(leftover)
    return order


class WorkflowExecutor(QObject):
    """Runs node handlers in topological order; supports async finish_node()."""

    node_started = pyqtSignal(str)
    node_finished = pyqtSignal(str, bool, str)  # key, ok, message
    pipeline_started = pyqtSignal()
    pipeline_finished = pyqtSignal(bool, str)  # ok, summary
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._handlers: dict[str, NodeHandler] = {}
        self._queue: list[str] = []
        self._index = 0
        self._running = False
        self._stop_requested = False
        self._current: Optional[str] = None
        self._ok_count = 0
        self._fail_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def set_handler(self, key: str, handler: NodeHandler) -> None:
        self._handlers[key] = handler

    def stop(self) -> None:
        self._stop_requested = True
        self.log.emit("已请求停止工作流…")

    def start(self, node_keys: list[str], edges: list[tuple[str, str]]) -> None:
        if self._running:
            self.log.emit("工作流已在运行")
            return
        order = topological_order(node_keys, edges)
        if not order:
            self.pipeline_finished.emit(False, "没有可执行的节点")
            return
        self._queue = order
        self._index = 0
        self._running = True
        self._stop_requested = False
        self._ok_count = 0
        self._fail_count = 0
        self._current = None
        self.pipeline_started.emit()
        self.log.emit("开始执行: " + " → ".join(order))
        self._advance()

    def finish_node(self, key: str, ok: bool, message: str = "") -> None:
        """Handlers must call this when a node completes (sync or async)."""
        if not self._running or self._current != key:
            return
        if ok:
            self._ok_count += 1
        else:
            self._fail_count += 1
        self.node_finished.emit(key, ok, message)
        self.log.emit(("✓ " if ok else "✗ ") + f"{key}: {message or ('完成' if ok else '失败')}")
        self._current = None
        self._index += 1
        if not ok:
            # Stop pipeline on failure
            self._finish_pipeline(False, f"在节点「{key}」失败: {message}")
            return
        self._advance()

    def _advance(self) -> None:
        if self._stop_requested:
            self._finish_pipeline(False, "用户停止")
            return
        if self._index >= len(self._queue):
            self._finish_pipeline(True, f"全部完成 ({self._ok_count} 成功)")
            return
        key = self._queue[self._index]
        handler = self._handlers.get(key)
        self._current = key
        self.node_started.emit(key)
        self.log.emit(f"▶ 执行节点: {key}")
        if handler is None:
            self.finish_node(key, False, "未注册执行器")
            return
        try:
            handler(key, self)
        except Exception as exc:  # noqa: BLE001 — surface to workflow UI
            self.finish_node(key, False, str(exc))

    def _finish_pipeline(self, ok: bool, summary: str) -> None:
        self._running = False
        self._current = None
        self.pipeline_finished.emit(ok, summary)
        self.log.emit(("工作流完成: " if ok else "工作流结束: ") + summary)
