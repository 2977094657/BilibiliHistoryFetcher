import platform
import psutil
import logging

logger = logging.getLogger(__name__)

# 最小资源要求
MIN_MEMORY_GB = 4  # 最小内存要求（GB）
MIN_FREE_DISK_GB = 2  # 最小可用磁盘空间（GB）
MIN_CPU_CORES = 2  # 最小CPU核心数

# 推荐资源要求
REC_MEMORY_GB = 8  # 推荐内存（GB）
REC_FREE_DISK_GB = 5  # 推荐可用磁盘空间（GB）
REC_CPU_CORES = 4  # 推荐CPU核心数

def check_system_resources():
    """
    检查系统资源是否满足运行语音转文字模型的要求
    
    返回:
        dict: 包含资源检查结果和详细信息的字典
    """
    try:
        # 获取系统信息
        os_name = platform.system()
        os_version = platform.version()
        
        # 检查内存
        total_memory_gb = psutil.virtual_memory().total / (1024**3)  # 转换为GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)  # 转换为GB
        
        # 检查CPU
        cpu_cores = psutil.cpu_count(logical=False)  # 物理核心数
        cpu_logical_cores = psutil.cpu_count(logical=True)  # 逻辑核心数
        cpu_usage = psutil.cpu_percent(interval=1)  # CPU使用率
        
        # 检查磁盘空间
        disk_usage = psutil.disk_usage('/')
        free_disk_gb = disk_usage.free / (1024**3)  # 转换为GB
        
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
                result["summary"]["resource_limitation"] = "可用内存不足"
            elif free_disk_gb < MIN_FREE_DISK_GB:
                result["summary"]["resource_limitation"] = "可用磁盘空间不足"
            elif not has_min_resources:
                result["summary"]["resource_limitation"] = "系统资源不满足最低要求"
        
        return result
    
    except Exception as e:
        logger.error(f"检查系统资源时出错: {str(e)}")
        # 返回一个保守的结果
        return {
            "error": str(e),
            "summary": {
                "has_minimum_resources": False,
                "has_recommended_resources": False,
                "can_run_speech_to_text": False,
                "resource_limitation": "检查资源时出错"
            }
        }
