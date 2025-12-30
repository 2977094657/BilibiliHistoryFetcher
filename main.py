import asyncio
import os
import platform
import sys
import traceback
import warnings
from contextlib import asynccontextmanager
from datetime import datetime

# 忽略jieba库中的无效转义序列警告
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from routers import (
    analysis,
    clean_data,
    export,
    fetch_bili_history,
    import_data_mysql,
    import_data_sqlite,
    heatmap,
    send_log,
    download,
    collection_download,
    history,
    categories,
    viewing_analytics,
    title_analytics,
    daily_count,
    login,
    delete_history,
    image_downloader,
    scheduler,
    video_summary,
    deepseek,
    audio_to_text,
    email_config,
    comment,
    data_sync,
    favorite,
    popular_videos,
    popular_analytics,
    bilibili_history_delete,
    video_details,
    dynamic
)
from scripts.scheduler_db_enhanced import EnhancedSchedulerDB
from scripts.scheduler_manager import SchedulerManager
from scripts.utils import load_config, get_output_path


# 配置日志系统
def setup_logging():
    """设置Loguru日志系统"""
    # 使用统一的日志初始化函数
    from scripts.utils import setup_logger

    # 初始化日志系统
    log_info = setup_logger()

    # 重定向 print 输出到日志
    class PrintToLogger:
        def __init__(self, stdout):
            self.stdout = stdout
            self._line_buffer = []
            self._is_shutting_down = False  # 标记系统是否正在关闭
            self._logging_in_progress = False  # 防止日志重入
            self._is_docker = os.environ.get('DOCKER_ENV') == 'true'  # 检测是否在Docker环境
            self._resource_warning_pattern = "系统资源不足"  # 用于识别资源警告的模式
            self._low_memory_detected = False  # 标记是否检测到内存不足

        def write(self, buf):
            # 如果系统正在关闭或在Docker环境中，直接写入原始stdout而不经过logger
            if self._is_shutting_down or self._is_docker:
                self.stdout.write(buf)
                return

            # 检查是否包含资源警告信息
            if self._resource_warning_pattern in buf:
                # 如果已经检测到内存不足，跳过重复的警告
                if self._low_memory_detected:
                    return
                # 标记已检测到内存不足
                self._low_memory_detected = True
                # 直接写入原始stdout，避免通过logger触发循环
                self.stdout.write(buf)
                return

            # 跳过uvicorn日志
            if any(skip in buf for skip in [
                'INFO:', 'ERROR:', 'WARNING:', '[32m', '[0m',
                'Application', 'Started', 'Waiting', 'HTTP',
                'uvicorn', 'DEBUG'
            ]):
                # 只写入到控制台
                self.stdout.write(buf)
                return

            # 检测是否是关闭信息
            if "应用关闭" in buf or "Shutting down" in buf:
                self._is_shutting_down = True

            # 防止日志重入 - 如果已经在记录日志中，直接写入原始stdout
            if self._logging_in_progress:
                self.stdout.write(buf)
                return

            # 收集完整的行
            for c in buf:
                if c == '\n':
                    line = ''.join(self._line_buffer).rstrip()
                    if line:  # 只记录非空行
                        try:
                            # 设置日志记录锁
                            self._logging_in_progress = True
                            # 使用loguru记录，但保持控制台干净
                            if not self._is_shutting_down:
                                try:
                                    logger.opt(depth=1).log("INFO", line)
                                except Exception as e:
                                    # 记录失败，写入原始stdout
                                    self.stdout.write(f"日志记录失败: {e}\n")
                                    self.stdout.write(f"{line}\n")
                            else:
                                # 关闭阶段直接写入控制台
                                self.stdout.write(f"{line}\n")
                        except Exception as e:
                            # 如果记录失败，写入原始stdout
                            self.stdout.write(f"日志异常: {e}\n")
                            self.stdout.write(f"{line}\n")
                        finally:
                            # 释放日志记录锁
                            self._logging_in_progress = False
                    self._line_buffer = []
                else:
                    self._line_buffer.append(c)

        def flush(self):
            if self._line_buffer:
                line = ''.join(self._line_buffer).rstrip()
                if line:
                    # 检查是否包含资源警告信息
                    if self._resource_warning_pattern in line:
                        if not self._low_memory_detected:
                            self._low_memory_detected = True
                            self.stdout.write(f"{line}\n")
                        self._line_buffer = []
                        self.stdout.flush()
                        return

                    # 防止日志重入
                    if self._logging_in_progress:
                        self.stdout.write(f"{line}\n")
                    else:
                        try:
                            self._logging_in_progress = True
                            if not self._is_shutting_down:
                                try:
                                    logger.opt(depth=1).log("INFO", line)
                                except Exception:
                                    self.stdout.write(f"{line}\n")
                            else:
                                self.stdout.write(f"{line}\n")
                        except Exception:
                            self.stdout.write(f"{line}\n")
                        finally:
                            self._logging_in_progress = False
                    self._line_buffer = []
                self.stdout.flush()

        def isatty(self):
            return self.stdout.isatty()

        def fileno(self):
            return self.stdout.fileno()

        # 在应用关闭阶段调用，标记关闭状态
        def mark_shutdown(self):
            self._is_shutting_down = True

    # 保存原始的stdout并重定向
    original_stdout = sys.stdout
    sys.stdout = PrintToLogger(original_stdout)

    # 配置uvicorn日志与loguru集成
    # 拦截标准库logging
    import logging
    class InterceptHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self._is_docker = os.environ.get('DOCKER_ENV') == 'true'
            self._in_emit = False  # 防止循环调用的标志
            self._resource_warning_pattern = "系统资源不足"  # 用于识别资源警告的模式

        def emit(self, record):
            # 防止循环调用
            if self._in_emit:
                # 如果已经在处理日志，直接返回
                return

            # 检查是否是资源警告消息，如果是则直接使用原始stdout输出
            message = record.getMessage()
            if self._resource_warning_pattern in message:
                # 对于资源警告，直接使用原始stdout输出，避免循环
                print(f"[{record.levelname}] {message}")
                return

            # 在Docker环境中，简化处理以避免循环
            if self._is_docker:
                # 使用原始的logging输出，不进行重定向
                print(f"[{record.levelname}] {message}")
                return

            # 设置处理标志，防止循环
            self._in_emit = True

            try:
                # 获取对应的Loguru级别名称
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                # 获取调用者的文件名和行号
                frame, depth = sys._getframe(6), 6
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                # 记录到日志文件，但不输出到控制台
                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, message
                )
            finally:
                # 重置处理标志
                self._in_emit = False

    # 替换所有标准库的日志处理器
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    return log_info

# 在应用启动时调用
setup_logging()

# 检查系统资源（针对Linux系统）
is_linux = platform.system().lower() == "linux"
if is_linux:
    try:
        from scripts.system_resource_check import check_system_resources
        resources = check_system_resources()
        if not resources["summary"]["can_run_speech_to_text"]:
            limitation = resources.get("summary", {}).get("resource_limitation", "未知原因")
            logger.warning(f"警告: 系统资源不足，语音转文字功能将被禁用。限制原因: {limitation}")
            logger.info(f"系统信息: 内存: {resources['memory']['total_gb']}GB (可用: {resources['memory']['available_gb']}GB), "
                      f"CPU: {resources['cpu']['physical_cores']}核心, 磁盘可用空间: {resources['disk']['free_gb']}GB")
    except ImportError:
        logger.warning("警告: 未安装psutil模块，无法检查系统资源。如需使用语音转文字功能，请安装psutil: pip install psutil")
    except Exception as e:
        logger.warning(f"警告: 检查系统资源时出错: {str(e)}")

# 全局调度器实例
scheduler_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global scheduler_manager

    logger.info("正在启动应用...")

    try:
        # 初始化增强版数据库
        EnhancedSchedulerDB.get_instance()
        logger.info("已初始化增强版调度器数据库")

        # 初始化调度器
        scheduler_manager = SchedulerManager.get_instance(app)

        # 显示调度器配置
        logger.info(f"调度器配置:")
        logger.info(f"  基础URL: {scheduler_manager.base_url}")
        logger.info("  (从 config/config.yaml 的 server 配置生成；当 host 为 0.0.0.0/:: 时自动使用 127.0.0.1)")

        # 检查基础URL是否包含协议前缀
        if not scheduler_manager.base_url.startswith(('http://', 'https://')):
            logger.warning(f"  警告: 基础URL不包含协议前缀，这可能导致任务执行错误")
            logger.warning(f"  建议: 检查服务器配置确保正确构建URL")

        # 创建异步任务运行调度器
        scheduler_task = asyncio.create_task(scheduler_manager.run_scheduler())

        # 加载配置并决定是否执行数据完整性校验
        current_config = load_config()
        check_on_startup = current_config.get('server', {}).get('data_integrity', {}).get('check_on_startup', True)
        if check_on_startup:
            logger.info("正在执行启动时数据完整性校验...")
            try:
                from scripts.check_data_integrity import check_data_integrity
                result = check_data_integrity()
                if result["success"]:
                    if result["difference"] == 0:
                        logger.success("数据完整性校验通过，数据库和JSON文件记录数一致")
                    else:
                        logger.warning(f"数据完整性校验发现差异: {result['difference']} 条记录")
                        logger.info(f"详细报告已保存到 {result['report_file']}")
                else:
                    logger.error("数据完整性校验失败")
            except Exception as e:
                logger.error(f"执行数据完整性校验时出错: {str(e)}")
        else:
            logger.info("已跳过启动时数据完整性校验")

        logger.success("=== 应用启动完成 ===")
        logger.info(f"启动时间: {datetime.now().isoformat()}")

        yield

        # 关闭时
        logger.info("\n=== 应用关闭阶段 ===")
        logger.info(f"开始时间: {datetime.now().isoformat()}")

        # 标记stdout重定向器为关闭状态，防止重入
        if hasattr(sys.stdout, 'mark_shutdown'):
            sys.stdout.mark_shutdown()

        if scheduler_manager:
            logger.info("正在停止调度器...")
            scheduler_manager.stop_scheduler()
            # 取消调度器任务
            scheduler_task.cancel()
            try:
                logger.info("等待调度器任务完成...")
                await scheduler_task
            except asyncio.CancelledError:
                logger.info("调度器任务已取消")

        # 恢复原始的 stdout
        if hasattr(sys.stdout, 'stdout'):
            logger.info("正在恢复标准输出...")
            sys.stdout = sys.stdout.stdout

        logger.success("=== 应用关闭完成 ===")
        logger.info(f"结束时间: {datetime.now().isoformat()}")

        # 清理日志处理器，防止关闭时的死锁
        # 注意：必须在所有日志记录之后调用
        try:
            print("正在安全关闭日志系统...")
            # 复制处理器列表，避免在迭代过程中修改
            logger_handlers = list(logger._core.handlers.keys())
            # 设置最小日志级别为ERROR，避免继续记录非关键日志
            for handler_id in logger_handlers:
                try:
                    # 修改处理器的最小记录级别为ERROR (40)
                    handler = logger._core.handlers.get(handler_id)
                    if handler:
                        handler._levelno = 40  # ERROR级别
                except Exception:
                    # 忽略修改失败的情况
                    pass
            # 逐个移除处理器
            for handler_id in logger_handlers:
                try:
                    logger.remove(handler_id)
                except Exception as log_ex:
                    print(f"移除日志处理器时出错 (忽略): {log_ex}")
            print("日志系统已安全关闭")
        except Exception as e:
            print(f"关闭日志系统时出错 (忽略): {e}")

    except Exception as e:
        logger.error(f"\n=== 应用生命周期出错 ===")
        logger.error(f"错误信息: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        # Loguru会自动提供详细堆栈
        logger.exception("应用生命周期异常")
        raise

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Bilibili History Analyzer",
    description="一个用于分析和导出Bilibili观看历史的API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加启动状态端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "scheduler_status": "running" if scheduler_manager and scheduler_manager.is_running else "stopped"
    }

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 注册路由
app.include_router(login.router, prefix="/login", tags=["用户登录"])
app.include_router(analysis.router, prefix="/analysis", tags=["数据分析"])
app.include_router(clean_data.router, prefix="/clean", tags=["数据清洗"])
app.include_router(export.router, prefix="/export", tags=["数据导出"])
app.include_router(fetch_bili_history.router, prefix="/fetch", tags=["历史记录获取"])
app.include_router(import_data_mysql.router, prefix="/importMysql", tags=["MySQL数据导入"])
app.include_router(import_data_sqlite.router, prefix="/importSqlite", tags=["SQLite数据导入"])
app.include_router(heatmap.router, prefix="/heatmap", tags=["热力图生成"])
app.include_router(send_log.router, prefix="/log", tags=["日志发送"])
app.include_router(download.router, prefix="/download", tags=["视频下载"])
app.include_router(collection_download.router, prefix="/collection", tags=["合集下载"])
app.include_router(history.router, prefix="/history", tags=["历史记录管理"])
app.include_router(categories.router, prefix="/categories", tags=["分类管理"])
app.include_router(viewing_analytics.router, prefix="/viewing", tags=["观看时间分析"])
app.include_router(title_analytics.router, prefix="/title", tags=["标题分析"])
app.include_router(daily_count.router, prefix="/daily", tags=["每日观看统计"])
app.include_router(delete_history.router, prefix="/delete", tags=["删除历史记录"])
app.include_router(image_downloader.router, prefix="/images", tags=["图片下载管理"])
app.include_router(scheduler.router, prefix="/scheduler", tags=["计划任务管理"])
app.include_router(video_summary.router, prefix="/summary", tags=["视频摘要"])
app.include_router(deepseek.router, prefix="/deepseek", tags=["DeepSeek AI"])
app.include_router(audio_to_text.router, prefix="/audio_to_text", tags=["音频转文字"])
app.include_router(email_config.router, prefix="/config", tags=["配置管理"])
app.include_router(comment.router, prefix="/comment", tags=["评论管理"])
app.include_router(data_sync.router, prefix="/data_sync", tags=["数据同步与完整性检查"])
app.include_router(favorite.router, prefix="/favorite", tags=["收藏夹管理"])
app.include_router(popular_videos.router, prefix="/bilibili", tags=["B站热门"])
app.include_router(popular_analytics.router, prefix="/popular", tags=["热门视频分析"])
app.include_router(bilibili_history_delete.router, prefix="/bilibili/history", tags=["B站历史记录删除"])
app.include_router(video_details.router, prefix="/video_details", tags=["视频详情"])
app.include_router(dynamic.router, prefix="/dynamic", tags=["用户动态"])

# 挂载静态目录，提供 output 下资源的访问（/static/ 相对路径）
try:
    static_dir = os.path.dirname(get_output_path("__base__"))
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"已挂载静态目录: /static -> {static_dir}")
except Exception as e:
    logger.warning(f"静态目录挂载失败（忽略）: {e}")

# 入口点，启动应用
if __name__ == "__main__":
    import uvicorn
    import atexit
    import signal

    # 配置日志系统
    log_info = setup_logging()

    # 标记是否正在关闭
    is_shutting_down = False

    # 信号处理函数
    def signal_handler(sig, frame):
        global is_shutting_down
        if is_shutting_down:
            print("正在关闭中，请稍候...")
            return
        is_shutting_down = True
        print(f"\n接收到信号 {sig}，正在优雅关闭...")
        # 标记stdout为关闭状态
        if hasattr(sys.stdout, 'mark_shutdown'):
            sys.stdout.mark_shutdown()
        sys.exit(0)

    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    # 注册退出时清理函数
    @atexit.register
    def cleanup_at_exit():
        global is_shutting_down
        if is_shutting_down:
            return
        is_shutting_down = True
        print("程序退出，正在清理资源...")
        # 恢复原始stdout
        if hasattr(sys.stdout, 'stdout'):
            sys.stdout = sys.stdout.stdout
        # 移除所有日志处理器
        print("正在关闭日志系统...")
        try:
            handlers = list(logger._core.handlers.keys())
            for handler_id in handlers:
                try:
                    logger.remove(handler_id)
                except Exception as e:
                    print(f"移除日志处理器时出错 (忽略): {e}")
        except Exception as e:
            print(f"获取日志处理器时出错 (忽略): {e}")
        print("日志系统已关闭")

    # 加载配置
    config = load_config()
    server_config = config.get('server', {})

    # 检查是否启用SSL
    ssl_enabled = server_config.get('ssl_enabled', False)
    ssl_certfile = server_config.get('ssl_certfile', None)
    ssl_keyfile = server_config.get('ssl_keyfile', None)

    # 使用SSL证书启动应用（如果启用）
    if ssl_enabled and ssl_certfile and ssl_keyfile:
        logger.info(f"使用HTTPS启动服务，端口: {server_config.get('port', 8899)}")
        logger.info(f"SSL证书路径: {ssl_certfile}")
        logger.info(f"SSL密钥路径: {ssl_keyfile}")
        try:
            # 检查证书文件是否存在
            if not os.path.exists(ssl_certfile):
                logger.error(f"错误: SSL证书文件不存在: {ssl_certfile}")
                sys.exit(1)

            if not os.path.exists(ssl_keyfile):
                logger.error(f"错误: SSL密钥文件不存在: {ssl_keyfile}")
                sys.exit(1)

            # 检查文件权限
            logger.info(f"证书文件权限: {oct(os.stat(ssl_certfile).st_mode)[-3:]}")
            logger.info(f"密钥文件权限: {oct(os.stat(ssl_keyfile).st_mode)[-3:]}")

            uvicorn.run(
                "main:app",
                host=server_config.get('host', "0.0.0.0"),  # 默认使用0.0.0.0允许所有IP访问
                port=server_config.get('port', 8899),
                log_level="debug",  # 修改为debug级别
                reload=False,  # 禁用热重载以避免多个调度器实例
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile
            )
        except Exception as e:
            logger.error(f"启动服务时出错: {e}")
            traceback.print_exc()
    else:
        logger.info(f"使用HTTP启动服务，主机: {server_config.get('host', '0.0.0.0')}，端口: {server_config.get('port', 8899)}")
        uvicorn.run(
            "main:app",
            host=server_config.get('host', "0.0.0.0"),  # 默认使用0.0.0.0允许所有IP访问
            port=server_config.get('port', 8899),
            log_level="info",
            reload=False  # 禁用热重载以避免多个调度器实例
        )
