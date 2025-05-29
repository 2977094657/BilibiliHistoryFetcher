import asyncio
import io
import sys
from importlib.metadata import entry_points
from typing import AsyncGenerator


# def _get_yutto_main():
#     """读取 console_scripts 的入口，返回真正的 main 函数"""
#     eps = entry_points(group="console_scripts")
#     if isinstance(eps, dict):               # 3.10+
#         yutto_ep = eps["yutto"][0]
#     else:                                   # 3.8 / 3.9
#         yutto_ep = next(e for e in eps if e.name == "yutto")
#     return yutto_ep.load()                  # 如 yutto.__main__.main


# # 提前加载一次，避免每次调用都解析 entry_points
# _YUTTO_MAIN = _get_yutto_main()

from yutto.__main__ import main as _YUTTO_MAIN


class _AsyncWriter(io.StringIO):
    """自定义的 StringIO：每次 write 时立即通过 Queue 推送到事件循环"""
    def __init__(self, queue: asyncio.Queue[str | None], loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._queue = queue
        self._loop = loop

    def write(self, s: str) -> int:
        """重写 write，每写一次就把内容丢到 Queue"""
        if s:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, s)
        return len(s)


async def run_yutto(argv: list[str]) -> AsyncGenerator[str, None]:
    """在当前进程内执行 yutto CLI，实时产出 SSE 数据"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    # 临时接管 stdout / stderr
    stdout_backup, stderr_backup = sys.stdout, sys.stderr
    sys.stdout = _AsyncWriter(queue, loop)
    sys.stderr = _AsyncWriter(queue, loop)

    # 在线程池执行同步的 yutto.main
    def _worker():
        # 伪装 sys.argv
        argv_backup = sys.argv
        sys.argv = ["yutto", *argv]
        try:
            _YUTTO_MAIN()                   # 进入 yutto 的主函数
        except SystemExit:                  # yutto 内部可能调用 sys.exit()
            pass
        finally:
            sys.argv = argv_backup
            # 通知协程：任务结束
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # 把 _worker 丢进默认线程池，避免阻塞事件循环
    loop.run_in_executor(None, _worker)

    # 异步迭代 queue，并包装成 SSE
    while True:
        line = await queue.get()
        if line is None:                    # 收到结束标记
            break
        yield f"data: {line.rstrip()}\n\n"

    # 恢复 stdout / stderr
    sys.stdout, sys.stderr = stdout_backup, stderr_backup
