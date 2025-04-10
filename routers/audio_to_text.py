"""
音频转文字API路由
处理视频语音转文字的API接口
"""

import os
from pathlib import Path
import time
import asyncio
import signal
import logging
import traceback
from typing import Optional, List, Dict, Any, TypedDict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json

from scripts.utils import load_config
from .whisper import WhisperModel
from huggingface_hub import snapshot_download, try_to_load_from_cache, scan_cache_dir

# 设置日志格式
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建API路由
router = APIRouter()
config = load_config()

# 全局变量
whisper_model = None
model_loading = False
model_lock = asyncio.Lock()


# 添加信号处理
def handle_interrupt(*_):
    """处理中断信号"""
    global whisper_model
    print("\n正在清理资源...")
    try:
        if whisper_model is not None:
            del whisper_model
            whisper_model = None
        print("资源已清理")
    except Exception as e:
        print(f"清理资源时出错: {str(e)}")
    # 不再调用 os._exit(0)，让服务继续运行


# 注册信号处理器
signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)


# 定义请求和响应模型
class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="音频文件路径，可以是相对路径或绝对路径")
    model_size: str = Field(
        "tiny",
        description="模型大小，可选值: tiny, base, small, medium, large-v1, large-v2, large-v3",
    )
    language: str = Field("zh", description="语言代码，默认为中文")
    cid: int = Field(..., description="视频的CID，用于分类存储和命名结果")


class TranscribeResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="处理结果或错误信息")
    duration: Optional[float] = Field(None, description="音频时长(秒)")
    processing_time: Optional[float] = Field(None, description="处理时间(秒)")
    language_detected: Optional[str] = Field(None, description="检测到的语言")
    cid: Optional[int] = Field(None, description="处理时使用的CID")


class SystemInfo(BaseModel):
    os_name: str = Field(..., description="操作系统名称")
    os_version: str = Field(..., description="操作系统版本")
    python_version: str = Field(..., description="Python版本")
    cuda_available: bool = Field(..., description="是否支持CUDA")
    cuda_version: Optional[str] = Field(None, description="CUDA版本")
    gpu_info: Optional[List[Dict[str, str]]] = Field(None, description="GPU信息")
    cuda_setup_guide: Optional[str] = Field(None, description="CUDA安装指南")
    resource_limitation: Optional[str] = Field(None, description="资源限制原因")


class ModelInfo(BaseModel):
    model_size: str = Field(..., description="模型大小")
    is_downloaded: bool = Field(..., description="模型是否已下载")
    model_path: Optional[str] = Field(None, description="模型文件路径")
    download_link: Optional[str] = Field(None, description="模型下载链接")
    file_size: Optional[str] = Field(None, description="模型文件大小")


class EnvironmentCheckResponse(BaseModel):
    system_info: SystemInfo
    models_info: Dict[str, ModelInfo]
    recommended_device: str = Field(..., description="推荐使用的设备(cuda/cpu)")
    compute_type: str = Field(..., description="推荐的计算类型(float16/int8)")


class WhisperModelInfo(BaseModel):
    """Whisper模型信息"""

    name: str = Field(..., description="模型名称")
    description: str = Field(..., description="模型描述")
    is_downloaded: bool = Field(..., description="是否已下载")
    path: Optional[str] = Field(None, description="模型路径")
    params_size: str = Field(..., description="参数大小")
    recommended_use: str = Field(..., description="推荐使用场景")


class ResourceCheckResponse(BaseModel):
    """系统资源检查响应"""

    os_info: Dict[str, Any] = Field(..., description="操作系统信息")
    memory: Dict[str, Any] = Field(..., description="内存信息")
    cpu: Dict[str, Any] = Field(..., description="CPU信息")
    disk: Dict[str, Any] = Field(..., description="磁盘信息")
    summary: Dict[str, Any] = Field(..., description="资源检查总结")
    can_run_speech_to_text: bool = Field(..., description="是否可以运行语音转文字功能")
    limitation_reason: Optional[str] = Field(None, description="限制原因")


class DeleteModelRequest(BaseModel):
    model_size: str = Field(
        ...,
        description="要删除的模型大小，可选值: tiny, base, small, medium, large-v1, large-v2, large-v3",
    )


# TODO:模型的的管理,包括下载/删除/载入/弹出/推理,注意并发冲突


async def load_model(model_size) -> WhisperModel:
    """加载Whisper模型"""
    global whisper_model, model_loading

    try:
        # 检查是否已加载相同型号的模型
        if whisper_model is not None and whisper_model.model_size == model_size:
            logger.info(f"使用已加载的模型: {model_size}")
            return whisper_model

        # 检查模型是否已下载
        model_path = try_to_load_from_cache(
            f"csukuangfj/sherpa-onnx-whisper-{model_size}", f"{model_size}-tokens.txt"
        )
        if model_path is None:
            logger.error(f"模型 {model_size} 尚未下载")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "MODEL_NOT_DOWNLOADED",
                    "message": f"模型 {model_size} 尚未下载，请先通过 /audio_to_text/models 接口查看可用模型，并确保选择已下载的模型",
                    "model_size": model_size,
                },
            )

        # 如果其他进程正在加载模型，等待
        if model_loading:
            logger.info("其他进程正在加载模型，等待...")
            wait_start = time.time()
            while model_loading:
                # 添加超时检查
                if time.time() - wait_start > 300:  # 5分钟超时
                    raise HTTPException(
                        status_code=500, detail="等待模型加载超时，请稍后重试"
                    )
                await asyncio.sleep(1)
            if whisper_model is not None and whisper_model.model_size == model_size:
                return whisper_model

        model_loading = True
        start_time = time.time()
        logger.info(f"开始加载模型: {model_size}")

        try:

            def _ensure_str(value: str | Any | None) -> str:
                if value is None:
                    raise ValueError("Expected str or bytes, got None")
                if isinstance(value, (str)):
                    return value
                raise TypeError(f"Expected str or bytes, got {type(value)}")

            loop = asyncio.get_running_loop()
            whisper_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    _ensure_str(
                        try_to_load_from_cache(
                            f"csukuangfj/sherpa-onnx-whisper-{model_size}",
                            f"{model_size}-encoder.onnx",
                        )
                    ),
                    _ensure_str(
                        try_to_load_from_cache(
                            f"csukuangfj/sherpa-onnx-whisper-{model_size}",
                            f"{model_size}-decoder.onnx",
                        )
                    ),
                    _ensure_str(
                        try_to_load_from_cache(
                            f"csukuangfj/sherpa-onnx-whisper-{model_size}",
                            f"{model_size}-tokens.txt",
                        )
                    ),
                    model_size=model_size,
                ),
            )

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {load_time:.2f} 秒")

            return whisper_model

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"模型加载过程出错: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"模型加载过程出错: {str(e)}")
    finally:
        model_loading = False
        logger.info("模型加载状态已重置")


@router.delete("/models", summary="删除指定的Whisper模型")
async def delete_model(request: DeleteModelRequest):
    """
    删除指定的Whisper模型

    Args:
        request: 包含要删除的模型大小

    Returns:
        dict: 包含删除操作结果的信息
    """
    try:
        # 如果模型正在使用中，不允许删除
        global whisper_model
        if whisper_model is not None and whisper_model.model_size == request.model_size:
            return {
                "success": False,
                "message": f"模型 {request.model_size} 当前正在使用中，无法删除。请先关闭使用该模型的任务后再尝试删除。",
                "model_size": request.model_size,
            }

        # 删除模型文件
        try:
            revisions = get_revisions_by_repo_id(
                f"csukuangfj/sherpa-onnx-whisper-{request.model_size}"
            )
            scan_cache_dir().delete_revisions(*revisions).execute()

            logger.info(f"已成功删除模型: {request.model_size}")
            return {
                "success": True,
                "message": f"已成功删除模型: {request.model_size}",
                "model_size": request.model_size,
            }
        except Exception as e:
            logger.error(f"删除模型文件时出错: {str(e)}")
            return {
                "success": False,
                "message": f"删除模型文件时出错: {str(e)}",
                "model_size": request.model_size,
            }

    except Exception as e:
        logger.error(f"删除模型时出错: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除模型时出错: {str(e)}")


@router.post("/download_model", summary="下载指定的Whisper模型")
async def download_model(model_size: str):
    """
    下载指定的Whisper模型

    Args:
        model_size: 模型大小，可选值: tiny, base, small, medium, large-v1, large-v2, large-v3
    """
    try:
        # 创建临时的WhisperModel实例来触发下载
        # 注意：这里会阻塞直到下载完成
        logger.info(f"开始下载模型: {model_size}")
        start_time = time.time()

        # 使用线程执行器来避免阻塞
        # doc: https://huggingface.co/docs/huggingface_hub/main/en/guides/download
        loop = asyncio.get_event_loop()
        model_path = await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=f"csukuangfj/sherpa-onnx-whisper-{model_size}",
                endpoint="https://hf-mirror.com",
            ),
        )

        download_time = time.time() - start_time
        logger.info(f"模型下载完成，耗时: {download_time:.2f} 秒")

        return {
            "status": "success",
            "message": f"模型 {model_size} 下载完成",
            "model_path": model_path,
            "download_time": f"{download_time:.2f}秒",
        }

    except Exception as e:
        logger.error(f"模型下载失败: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"模型下载失败: {str(e)}")


def format_timestamp(seconds):
    """将秒转换为完整的时间戳格式 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)

    # 始终返回完整的 HH:MM:SS 格式
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


def save_transcript(all_segments: list[str], output_path: str) -> None:
    """保存转录结果为简洁格式，适合节省token"""
    print(f"准备保存转录结果到: {output_path}")
    print(f"处理的片段数量: {len(all_segments)}")

    # 整理数据：移除多余的空格和控制字符
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=4)

    print(f"转录结果已保存: {output_path}")


def get_revisions_by_repo_id(repo_id):
    cache_info = scan_cache_dir()
    revisions = []

    for i in cache_info.repos:
        if i.repo_id == repo_id:
            for r in i.revisions:
                revisions.append(r.commit_hash)

    return revisions


async def transcribe_audio(
    audio_path: Path, model_size: str = "medium", language="zh", cid=None
):
    """转录音频文件"""
    try:
        logger.info(f"开始处理音频文件: {audio_path}")
        logger.info(f"参数: model_size={model_size}, language={language}, cid={cid}")

        start_time = time.time()

        logger.info("准备加载模型...")
        model = await load_model(model_size)
        logger.info("模型加载完成")

        logger.info("开始转录音频...")
        segments, info = model.run(audio_path, language=language)
        logger.info("音频转录完成")

        # 处理结果
        logger.info("处理转录结果...")
        all_segments = list(segments)
        logger.info(f"转录得到 {len(all_segments)} 个片段")

        # 如果指定了CID，保存到对应目录
        if cid:
            logger.info(f"准备保存结果到CID目录: {cid}")
            save_dir = os.path.join("output", "stt", str(cid))
            os.makedirs(save_dir, exist_ok=True)

            # 保存JSON格式
            json_path = os.path.join(save_dir, f"{cid}.json")
            logger.info(f"保存JSON格式到: {json_path}")
            save_transcript(all_segments, json_path)
            logger.info("转录结果保存完成")

        processing_time = time.time() - start_time
        logger.info(f"总处理时间: {processing_time:.2f} 秒")

        return {
            "success": True,
            "message": "转录完成",
            "duration": info["duration"],
            "language_detected": info["language"],
            "processing_time": processing_time,
        }
    except Exception as e:
        logger.error(f"转录过程出错: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe", response_model=TranscribeResponse, summary="转录音频文件")
async def transcribe_audio_api(request: TranscribeRequest):
    """转录音频文件为文本"""
    try:
        start_time = time.time()
        logger.info(
            f"收到转录请求: {request.audio_path}, 模型: {request.model_size}, 语言: {request.language}, CID: {request.cid}"
        )

        audio_path = Path(request.audio_path)
        if not audio_path.exists():
            raise HTTPException(status_code=400, detail="Unable to find audio path")

        result = await transcribe_audio(
            audio_path=audio_path,
            model_size=request.model_size,
            language=request.language,
            cid=request.cid,
        )

        processing_time = time.time() - start_time
        logger.info(f"转录完成，耗时: {processing_time:.2f} 秒")

        return TranscribeResponse(
            success=result["success"],
            message=result["message"],
            duration=result.get("duration"),
            processing_time=processing_time,
            language_detected=result.get("language_detected"),
            cid=request.cid,
        )
    except Exception as e:
        logger.error(f"转录过程出错: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"转录过程出错: {str(e)}")


@router.get("/models", response_model=List[WhisperModelInfo])
async def list_models():
    """
    列出可用的Whisper模型，并显示每个模型的下载状态和详细信息

    Returns:
        模型列表，包含名称、描述、下载状态等信息
    """
    # 定义模型信息
    model_infos = [
        {
            "name": "tiny.en",
            "description": "极小型(英语专用)",
            "params_size": "39M参数",
            "recommended_use": "适用于简单的英语语音识别，对资源要求最低",
        },
        {
            "name": "base.en",
            "description": "基础型(英语专用)",
            "params_size": "74M参数",
            "recommended_use": "适用于一般的英语语音识别，速度和准确度均衡",
        },
        {
            "name": "small.en",
            "description": "小型(英语专用)",
            "params_size": "244M参数",
            "recommended_use": "适用于较复杂的英语语音识别，准确度较高",
        },
        {
            "name": "medium.en",
            "description": "中型(英语专用)",
            "params_size": "769M参数",
            "recommended_use": "适用于专业的英语语音识别，准确度高",
        },
        {
            "name": "tiny",
            "description": "极小型(多语言)",
            "params_size": "39M参数",
            "recommended_use": "适用于简单的多语言语音识别，特别是资源受限场景",
        },
        {
            "name": "base",
            "description": "基础型(多语言)",
            "params_size": "74M参数",
            "recommended_use": "适用于一般的多语言语音识别，平衡性能和资源占用",
        },
        {
            "name": "small",
            "description": "小型(多语言)",
            "params_size": "244M参数",
            "recommended_use": "适用于较复杂的多语言语音识别，准确度和性能均衡",
        },
        {
            "name": "medium",
            "description": "中型(多语言)",
            "params_size": "769M参数",
            "recommended_use": "适用于专业的多语言语音识别，高准确度",
        },
        {
            "name": "large-v1",
            "description": "大型V1",
            "params_size": "1550M参数",
            "recommended_use": "适用于要求极高准确度的场景，支持所有语言",
        },
        {
            "name": "large-v2",
            "description": "大型V2",
            "params_size": "1550M参数",
            "recommended_use": "V1的改进版本，提供更好的多语言支持",
        },
        {
            "name": "large-v3",
            "description": "大型V3",
            "params_size": "1550M参数",
            "recommended_use": "最新版本，提供最佳的识别效果和语言支持",
        },
    ]

    result = []
    for model_info in model_infos:
        model_path_encoder = try_to_load_from_cache(
            f"csukuangfj/sherpa-onnx-whisper-{model_info['name']}",
            f"{model_info['name']}-encoder.onnx",
        )
        model_path_decoder = try_to_load_from_cache(
            f"csukuangfj/sherpa-onnx-whisper-{model_info['name']}",
            f"{model_info['name']}-decoder.onnx",
        )
        model_path_token = try_to_load_from_cache(
            f"csukuangfj/sherpa-onnx-whisper-{model_info['name']}",
            f"{model_info['name']}-tokens.txt",
        )

        result.append(
            WhisperModelInfo(
                name=model_info["name"],
                description=model_info["description"],
                is_downloaded=True
                if model_path_encoder and model_path_decoder and model_path_token
                else False,
                path=os.path.dirname(model_path_encoder)
                if model_path_encoder
                else None,
                params_size=model_info["params_size"],
                recommended_use=model_info["recommended_use"],
            )
        )

    return result


class AudioPath(TypedDict):
    cid: int
    path: str


async def find_audio(cid: int) -> AudioPath:
    download_dir = Path("output") / "download_video"
    audio_files: list[Path] = []
    patterns = [f"**/*_{cid}.m4a", f"**/*_{cid}.mp4", f"**/*_{cid}.wav"]
    for p in patterns:
        audio_files.extend(download_dir.rglob(p))

    if len(audio_files) == 0:
        raise Exception(f"未找到CID为{cid}的音频文件")
    else:
        return AudioPath(cid=cid, path=str(audio_files[0]))


class FindAudioResponse(BaseModel):
    cid: int
    audio_path: str


@router.get(
    "/find_audio", summary="根据CID查找音频文件路径", response_model=FindAudioResponse
)
async def find_audio_by_cid(cid: int):
    """
    根据CID查找对应的音频文件路径

    Args:
        cid: 视频的CID

    Returns:
        音频文件的完整路径
    """
    try:
        audio_path = await find_audio(cid)
        return FindAudioResponse(cid=cid, audio_path=audio_path["path"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查找音频文件时出错: {str(e)}")


def get_cuda_setup_guide(os_name: str) -> str:
    """根据操作系统生成CUDA安装指南"""
    if os_name.lower() == "windows":
        return """Windows CUDA安装步骤：
1. 访问 NVIDIA 驱动下载页面：https://www.nvidia.cn/Download/index.aspx
2. 下载并安装适合您显卡的最新驱动
3. 访问 NVIDIA CUDA 下载页面：https://developer.nvidia.cn/cuda-downloads
4. 选择Windows版本下载并安装CUDA Toolkit
5. 安装完成后重启系统
6. 在命令行中输入 'nvidia-smi' 验证安装"""
    elif os_name.lower() == "linux":
        return """Linux CUDA安装步骤：
1. 检查系统是否有NVIDIA显卡：
   lspci | grep -i nvidia

2. 安装NVIDIA驱动：
   Ubuntu/Debian:
   sudo apt update
   sudo apt install nvidia-driver-xxx（替换xxx为最新版本号）

   CentOS:
   sudo dnf install nvidia-driver

3. 安装CUDA Toolkit：
   访问：https://developer.nvidia.com/cuda-downloads
   选择对应的Linux发行版，按照页面提供的命令安装

4. 设置环境变量：
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc

5. 验证安装：
   nvidia-smi
   nvcc --version"""
    else:
        return "暂不支持当前操作系统的CUDA安装指南"


@router.get("/check_environment", response_model=EnvironmentCheckResponse)
async def check_environment():
    """
    检查系统环境、CUDA支持情况

    返回：
    - 系统信息（操作系统、Python版本等）
    - CUDA支持情况
    - 推荐配置
    """
    try:
        import platform
        import sys

        # 获取系统信息
        os_name = platform.system()
        os_version = platform.version()
        python_version = sys.version

        # 检查CUDA支持
        cuda_available = False
        cuda_version = None
        gpu_info = None
        resource_limitation = None

        # 生成CUDA安装指南
        cuda_setup_guide = None if cuda_available else get_cuda_setup_guide(os_name)

        # 获取系统信息
        system_info = SystemInfo(
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_info=gpu_info,
            cuda_setup_guide=cuda_setup_guide,
            resource_limitation=resource_limitation,
        )

        # 确定推荐设备和计算类型
        recommended_device = "cuda" if cuda_available else "cpu"
        compute_type = "float16" if cuda_available else "int8"

        return EnvironmentCheckResponse(
            system_info=system_info,
            models_info={},  # 移除模型信息，因为已经在 /models 接口中提供
            recommended_device=recommended_device,
            compute_type=compute_type,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"环境检查失败: {str(e)}")


@router.get("/resource_check", response_model=ResourceCheckResponse)
async def check_system_resources_api():
    """
    检查系统资源是否满足运行语音转文字模型的要求

    返回：
    - 操作系统信息
    - 内存信息
    - CPU信息
    - 磁盘信息
    - 资源检查总结
    - 是否可以运行语音转文字功能
    - 限制原因（如果有）
    """
    try:
        from scripts.system_resource_check import check_system_resources

        resources = check_system_resources()

        # 添加语音转文字功能可用性信息
        can_run = resources["summary"]["can_run_speech_to_text"]
        limitation = resources.get("summary", {}).get("resource_limitation", None)

        # 构建响应
        response = ResourceCheckResponse(
            os_info=resources["os_info"],
            memory=resources["memory"],
            cpu=resources["cpu"],
            disk=resources["disk"],
            summary=resources["summary"],
            can_run_speech_to_text=can_run,
            limitation_reason=limitation,
        )

        return response
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="未安装psutil模块，无法检查系统资源。请安装psutil: pip install psutil",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查系统资源时出错: {str(e)}")


@router.get("/check_stt_file", summary="检查指定CID的转换后文件是否存在")
async def check_stt_file(cid: int):
    """
    检查指定CID的语音转文字文件是否存在

    Args:
        cid: 视频的CID

    Returns:
        dict: 包含文件是否存在的信息
    """
    try:
        # 构建文件路径
        save_dir = os.path.join("output", "stt", str(cid))
        json_path = os.path.join(save_dir, f"{cid}.json")

        # 检查文件是否存在
        exists = os.path.exists(json_path)

        return {
            "success": True,
            "exists": exists,
            "cid": cid,
            "file_path": json_path if exists else None,
        }

    except Exception as e:
        logger.error(f"检查STT文件时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检查STT文件时出错: {str(e)}")
