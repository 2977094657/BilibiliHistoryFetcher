import platform

try:
    import psutil
except ImportError:
    # 如果无法导入psutil，使用print而不是logger避免循环引用
    print("警告: 未安装psutil模块，无法检查系统资源。如需使用语音转文字功能，请安装psutil: pip install psutil")
    psutil = None

# 避免循环引用，使用print而不是logger
# 只有在明确需要记录到日志文件时才导入logger
# from loguru import logger

# 最小资源要求
MIN_MEMORY_GB = 4  # 最小内存要求（GB）
MIN_FREE_DISK_GB = 2  # 最小可用磁盘空间（GB）
MIN_CPU_CORES = 2  # 最小CPU核心数

# 推荐资源要求
REC_MEMORY_GB = 8  # 推荐内存（GB）
REC_FREE_DISK_GB = 5  # 推荐可用磁盘空间（GB）
REC_CPU_CORES = 4  # 推荐CPU核心数

# 缓存检查结果，避免重复检查
_cached_resource_check = None

def check_system_resources():
    """
    检查系统资源是否满足运行语音转文字模型的要求

    返回:
        dict: 包含资源检查结果和详细信息的字典
    """
    global _cached_resource_check

    # 如果已经检查过，直接返回缓存结果
    if _cached_resource_check is not None:
        return _cached_resource_check

    # 如果psutil未安装，返回保守结果
    if psutil is None:
        _cached_resource_check = {
            "error": "psutil模块未安装",
            "os_info": {
                "name": platform.system(),
                "version": platform.version(),
                "is_linux": platform.system().lower() == "linux"
            },
            "memory": {
                "total_gb": 0,
                "available_gb": 0,
                "meets_minimum": False,
                "meets_recommended": False
            },
            "cpu": {
                "physical_cores": 0,
                "logical_cores": 0,
                "usage_percent": 0,
                "meets_minimum": False,
                "meets_recommended": False
            },
            "disk": {
                "free_gb": 0,
                "meets_minimum": False,
                "meets_recommended": False
            },
            "summary": {
                "has_minimum_resources": False,
                "has_recommended_resources": False,
                "can_run_speech_to_text": False,
                "resource_limitation": "psutil模块未安装"
            }
        }
        return _cached_resource_check

    try:
        # 获取系统信息
        os_name = platform.system()
        os_version = platform.version()

        # 检查内存 - 使用异常处理避免在内存不足时崩溃
        try:
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)  # 转换为GB
            available_memory_gb = memory.available / (1024**3)  # 转换为GB
        except Exception as e:
            # 内存检查失败，使用保守估计
            print(f"内存检查失败: {str(e)}")
            total_memory_gb = 0
            available_memory_gb = 0

        # 检查CPU - 使用异常处理避免在CPU负载高时崩溃
        try:
            cpu_cores = psutil.cpu_count(logical=False)  # 物理核心数
            cpu_logical_cores = psutil.cpu_count(logical=True)  # 逻辑核心数
            # 减少CPU检查的时间间隔，避免长时间阻塞
            cpu_usage = psutil.cpu_percent(interval=0.1)  # CPU使用率
        except Exception as e:
            # CPU检查失败，使用保守估计
            print(f"CPU检查失败: {str(e)}")
            cpu_cores = 0
            cpu_logical_cores = 0
            cpu_usage = 100  # 假设CPU负载高

        # 检查磁盘空间 - 使用异常处理避免在磁盘问题时崩溃
        try:
            disk_usage = psutil.disk_usage('/')
            free_disk_gb = disk_usage.free / (1024**3)  # 转换为GB
        except Exception as e:
            # 磁盘检查失败，使用保守估计
            print(f"磁盘检查失败: {str(e)}")
            free_disk_gb = 0

        # 判断资源是否足够
        has_min_resources = (
            total_memory_gb >= MIN_MEMORY_GB and
            free_disk_gb >= MIN_FREE_DISK_GB and
            (cpu_cores or cpu_logical_cores) >= MIN_CPU_CORES
        )

        has_recommended_resources = (
            total_memory_gb >= REC_MEMORY_GB and
            free_disk_gb >= REC_FREE_DISK_GB and
            (cpu_cores or cpu_logical_cores) >= REC_CPU_CORES
        )

        # 准备结果
        result = {
            "os_info": {
                "name": os_name,
                "version": os_version,
                "is_linux": os_name.lower() == "linux"
            },
            "memory": {
                "total_gb": round(total_memory_gb, 2),
                "available_gb": round(available_memory_gb, 2),
                "meets_minimum": total_memory_gb >= MIN_MEMORY_GB,
                "meets_recommended": total_memory_gb >= REC_MEMORY_GB
            },
            "cpu": {
                "physical_cores": cpu_cores,
                "logical_cores": cpu_logical_cores,
                "usage_percent": cpu_usage,
                "meets_minimum": (cpu_cores or cpu_logical_cores) >= MIN_CPU_CORES,
                "meets_recommended": (cpu_cores or cpu_logical_cores) >= REC_CPU_CORES
            },
            "disk": {
                "free_gb": round(free_disk_gb, 2),
                "meets_minimum": free_disk_gb >= MIN_FREE_DISK_GB,
                "meets_recommended": free_disk_gb >= REC_FREE_DISK_GB
            },
            "summary": {
                "has_minimum_resources": has_min_resources,
                "has_recommended_resources": has_recommended_resources,
                "can_run_speech_to_text": has_min_resources
            }
        }

        # 如果是Linux系统，进行额外检查
        if os_name.lower() == "linux":
            # 检查是否有足够的内存用于语音转文字
            result["summary"]["can_run_speech_to_text"] = has_min_resources and available_memory_gb >= MIN_MEMORY_GB

            # 检查CPU负载
            if cpu_usage > 80:  # 如果CPU使用率超过80%
                result["summary"]["can_run_speech_to_text"] = False
                result["summary"]["resource_limitation"] = "CPU负载过高"
            elif available_memory_gb < MIN_MEMORY_GB:
                result["summary"]["can_run_speech_to_text"] = False
                result["summary"]["resource_limitation"] = "可用内存不足"
            elif free_disk_gb < MIN_FREE_DISK_GB:
                result["summary"]["can_run_speech_to_text"] = False
                result["summary"]["resource_limitation"] = "可用磁盘空间不足"
            elif not has_min_resources:
                result["summary"]["can_run_speech_to_text"] = False
                result["summary"]["resource_limitation"] = "系统资源不满足最低要求"

        # 缓存结果
        _cached_resource_check = result
        return result

    except Exception as e:
        # 使用print而不是logger避免循环引用
        print(f"检查系统资源时出错: {str(e)}")

        # 返回一个保守的结果
        result = {
            "error": str(e),
            "os_info": {
                "name": platform.system(),
                "version": platform.version(),
                "is_linux": platform.system().lower() == "linux"
            },
            "memory": {
                "total_gb": 0,
                "available_gb": 0,
                "meets_minimum": False,
                "meets_recommended": False
            },
            "cpu": {
                "physical_cores": 0,
                "logical_cores": 0,
                "usage_percent": 0,
                "meets_minimum": False,
                "meets_recommended": False
            },
            "disk": {
                "free_gb": 0,
                "meets_minimum": False,
                "meets_recommended": False
            },
            "summary": {
                "has_minimum_resources": False,
                "has_recommended_resources": False,
                "can_run_speech_to_text": False,
                "resource_limitation": "检查资源时出错"
            }
        }

        # 缓存结果
        _cached_resource_check = result
        return result

def can_import_faster_whisper():
    """
    检查是否可以导入faster_whisper模块

    返回:
        bool: 如果可以导入faster_whisper则返回True，否则返回False
    """
    try:
        # 检查系统资源
        resources = check_system_resources()

        # 如果是Linux系统且资源不足，则不导入faster_whisper
        if resources["os_info"]["is_linux"] and not resources["summary"]["can_run_speech_to_text"]:
            print(f"系统资源不足，不导入faster_whisper模块。限制原因: {resources.get('summary', {}).get('resource_limitation', '未知')}")
            return False

        # 尝试导入faster_whisper
        try:
            import faster_whisper
            return True
        except ImportError:
            print("无法导入faster_whisper模块")
            return False
    except ImportError:
        print("无法导入faster_whisper模块")
        return False
    except Exception as e:
        print(f"检查faster_whisper导入时出错: {str(e)}")
        return False

# 保留原函数以保持兼容性，但实际上现在我们不再使用torch
def can_import_torch():
    """
    检查是否可以导入torch模块 (为了兼容性而保留，实际使用can_import_faster_whisper)

    返回:
        bool: 调用can_import_faster_whisper的结果
    """
    return can_import_faster_whisper()
