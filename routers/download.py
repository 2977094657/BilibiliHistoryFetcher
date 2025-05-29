import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import httpx
import json

from scripts.utils import load_config
from scripts.yutto_runner import run_yutto

# 尝试导入 history 模块，用于处理图像 URL
try:
    from routers.history import _process_image_url, get_video_by_cid, _process_record
except ImportError:
    # 如果无法直接导入，则在运行时动态加载
    _process_image_url = None
    get_video_by_cid = None
    _process_record = None
    print("无法从 history 模块导入_process_image_url、get_video_by_cid 和_process_record 函数")

router = APIRouter()
config = load_config()



# 辅助函数：检查下载目录和临时目录
def check_download_directories() -> Tuple[str, str]:
    """
    检查下载目录和临时目录是否存在且有写入权限

    Returns:
        下载目录和临时目录的路径元组

    Raises:
        HTTPException: 如果目录不存在或没有写入权限
    """
    download_dir = os.path.normpath(config['yutto']['basic']['dir'])
    tmp_dir = os.path.normpath(config['yutto']['basic']['tmp_dir'])

    # 创建目录（如果不存在）
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # 检查目录权限
    if not os.access(download_dir, os.W_OK):
        raise HTTPException(
            status_code=500,
            detail=f"没有下载目录的写入权限：{download_dir}"
        )
    if not os.access(tmp_dir, os.W_OK):
        raise HTTPException(
            status_code=500,
            detail=f"没有临时目录的写入权限：{tmp_dir}"
        )

    return download_dir, tmp_dir

# 辅助函数：准备进程参数
def prepare_process_kwargs() -> Dict[str, Any]:
    """
    准备进程参数

    Returns:
        进程参数字典
    """
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'  # 确保 Python 输出不被缓存

    # 在 Linux 上确保 PATH 包含 python 环境
    if sys.platform != 'win32':
        env['PATH'] = f"{os.path.dirname(sys.executable)}:{env.get('PATH', '')}"
        # 添加 virtualenv 的 site-packages 路径（如果存在）
        site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'lib', 'python*/site-packages')
        env['PYTHONPATH'] = f"{site_packages}:{env.get('PYTHONPATH', '')}"

    # 准备进程参数
    popen_kwargs = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'encoding': 'utf-8',
        'errors': 'replace',
        'universal_newlines': True,
        'env': env,
        'bufsize': 1,  # 行缓冲
        'shell': sys.platform != 'win32'  # 在非 Windows 系统上使用 shell
    }

    # 在 Windows 系统上添加 CREATE_NO_WINDOW 标志
    if sys.platform == 'win32':
        popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        popen_kwargs['shell'] = False  # Windows 上不使用 shell

    return popen_kwargs

# 辅助函数：格式化命令
def format_command(command: List[str]) -> str:
    """
    格式化命令，处理特殊字符

    Args:
        command: 命令列表

    Returns:
        格式化后的命令字符串
    """
    if sys.platform != 'win32':
        return ' '.join(f"'{arg}'" if ((' ' in arg) or ("'" in arg) or ('"' in arg)) else arg for arg in command)
    return ' '.join(command)

# 辅助函数：添加下载参数到命令
def add_download_params_to_command(command: List[str], params: Any) -> List[str]:
    """
    将下载参数添加到命令中

    Args:
        command: 命令列表
        params: 包含下载参数的对象

    Returns:
        添加了下载参数的命令列表
    """
    # 基础参数
    # 视频清晰度
    if params.video_quality is not None:
        command.extend(['--video-quality', str(params.video_quality)])

    # 音频码率
    if params.audio_quality is not None:
        command.extend(['--audio-quality', str(params.audio_quality)])

    # 视频编码
    if params.vcodec:
        command.extend(['--vcodec', params.vcodec])

    # 音频编码
    if params.acodec:
        command.extend(['--acodec', params.acodec])

    # 视频下载编码优先级
    if params.download_vcodec_priority:
        command.extend(['--download-vcodec-priority', params.download_vcodec_priority])

    # 输出格式
    if params.output_format:
        command.extend(['--output-format', params.output_format])

    # 仅包含音频流时的输出格式
    if params.output_format_audio_only:
        command.extend(['--output-format-audio-only', params.output_format_audio_only])

    # 资源选择参数
    # 仅下载视频流
    if params.video_only:
        command.append('--video-only')

    # 仅下载音频流
    if params.only_audio:
        command.append('--audio-only')

    # 不生成弹幕文件
    if params.no_danmaku:
        command.append('--no-danmaku')

    # 仅生成弹幕文件
    if params.danmaku_only:
        command.append('--danmaku-only')

    # 不生成字幕文件
    if params.no_subtitle or not config['yutto']['resource']['require_subtitle']:
        command.append('--no-subtitle')

    # 仅生成字幕文件
    if params.subtitle_only:
        command.append('--subtitle-only')

    # 仅生成媒体元数据文件
    if params.metadata_only:
        command.append('--metadata-only')

    # 不生成视频封面
    if not params.download_cover:
        command.append('--no-cover')

    # 生成视频流封面时单独保存封面
    if params.save_cover:
        command.append('--save-cover')

    # 仅生成视频封面
    if params.cover_only:
        command.append('--cover-only')

    # 不生成章节信息
    if params.no_chapter_info:
        command.append('--no-chapter-info')

    # 添加其他 yutto 配置
    if config['yutto']['danmaku']['font_size']:
        command.extend(['--danmaku-font-size', str(config['yutto']['danmaku']['font_size'])])

    if config['yutto']['batch']['with_section']:
        command.append('--with-section')

    # 如果提供了 SESSDATA，添加到命令中
    if hasattr(params, 'sessdata') and params.sessdata:
        command.extend(['--sessdata', params.sessdata])
    elif config.get('SESSDATA'):
        command.extend(['--sessdata', config['SESSDATA']])

    return command

def extract_datetime_from_string(text):
    """
    从字符串中提取日期时间

    支持的格式：
    1. YYYYMMDD_HHMMSS
    2. YYYYMMDD_HHMM
    3. YYYYMMDD
    4. Unix 时间戳

    Args:
        text: 要检查的字符串

    Returns:
        格式化的日期时间字符串或 None
    """
    # 调试信息
    print(f"【时间提取】尝试从'{text}'中提取日期时间")

    # 尝试匹配 YYYYMMDD_HHMMSS 格式
    match1 = re.match(r'.*?(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2}).*', text)
    if match1:
        year, month, day, hour, minute, second = match1.groups()
        result = f"{year}-{month}-{day} {hour}:{minute}:{second}"
        print(f"【时间提取】匹配 YYYYMMDD_HHMMSS 格式：{result}")
        return result

    # 尝试匹配 YYYYMMDD_HHMM 格式
    match2 = re.match(r'.*?(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2}).*', text)
    if match2:
        year, month, day, hour, minute = match2.groups()
        result = f"{year}-{month}-{day} {hour}:{minute}:00"
        print(f"【时间提取】匹配 YYYYMMDD_HHMM 格式：{result}")
        return result

    # 尝试匹配纯 YYYYMMDD 格式
    match3 = re.match(r'.*?(\d{4})(\d{2})(\d{2}).*', text)
    if match3:
        year, month, day = match3.groups()
        result = f"{year}-{month}-{day} 00:00:00"
        print(f"【时间提取】匹配 YYYYMMDD 格式：{result}")
        return result

    # 尝试匹配 Unix 时间戳（最后 10 位数字）
    match4 = re.match(r'^(\d{10})$', text)
    if match4:
        try:
            timestamp = int(match4.group(1))
            dt = datetime.fromtimestamp(timestamp)
            result = dt.strftime("%Y-%m-%d %H:%M:%S")
            print(f"【时间提取】匹配 Unix 时间戳：{result}")
            return result
        except:
            pass

    print(f"【时间提取】未能从'{text}'中提取日期时间")
    return None

# 基础下载参数模型类
class BaseDownloadParams(BaseModel):
    """所有下载请求的基础参数"""
    sessdata: Optional[str] = Field(None, description="用户的 SESSDATA")
    download_cover: Optional[bool] = Field(True, description="是否下载视频封面")
    only_audio: Optional[bool] = Field(False, description="是否仅下载音频")

    # 基础参数
    video_quality: Optional[int] = Field(None, description="视频清晰度等级，可选值：127|126|125|120|116|112|100|80|74|64|32|16")
    audio_quality: Optional[int] = Field(None, description="音频码率等级，可选值：30251|30255|30250|30280|30232|30216")
    vcodec: Optional[str] = Field(None, description="视频编码，格式为'下载编码:保存编码'，如'avc:copy'")
    acodec: Optional[str] = Field(None, description="音频编码，格式为'下载编码:保存编码'，如'mp4a:copy'")
    download_vcodec_priority: Optional[str] = Field(None, description="视频下载编码优先级，如'hevc,avc,av1'")
    output_format: Optional[str] = Field(None, description="输出格式，可选值：infer|mp4|mkv|mov")
    output_format_audio_only: Optional[str] = Field(None, description="仅包含音频流时的输出格式，可选值：infer|m4a|aac|mp3|flac|mp4|mkv|mov")

    # 资源选择参数
    video_only: Optional[bool] = Field(False, description="是否仅下载视频流")
    danmaku_only: Optional[bool] = Field(False, description="是否仅生成弹幕文件")
    no_danmaku: Optional[bool] = Field(False, description="是否不生成弹幕文件")
    subtitle_only: Optional[bool] = Field(False, description="是否仅生成字幕文件")
    no_subtitle: Optional[bool] = Field(False, description="是否不生成字幕文件")
    metadata_only: Optional[bool] = Field(False, description="是否仅生成媒体元数据文件")
    save_cover: Optional[bool] = Field(False, description="生成视频流封面时是否单独保存封面")
    cover_only: Optional[bool] = Field(False, description="是否仅生成视频封面")
    no_chapter_info: Optional[bool] = Field(False, description="是否不生成章节信息")

class DownloadRequest(BaseDownloadParams):
    """单个视频下载请求"""
    url: str = Field(..., description="视频URL")
    cid: int = Field(..., description="视频的 CID，用于分类存储和音频文件命名前缀")

class UserSpaceDownloadRequest(BaseDownloadParams):
    """用户空间视频下载请求"""
    user_id: str = Field(..., description="用户 UID，例如：100969474")

class FavoriteDownloadRequest(BaseDownloadParams):
    """收藏夹下载请求"""
    user_id: str = Field(..., description="用户 UID，例如：100969474")
    fid: Optional[str] = Field(None, description="收藏夹 ID，不提供时下载所有收藏夹")

class VideoInfo(BaseModel):
    """视频信息"""
    bvid: str = Field(..., description="视频的 BVID")
    cid: int = Field(..., description="视频的 CID")
    title: Optional[str] = Field(None, description="视频标题")
    author: Optional[str] = Field(None, description="视频作者")
    cover: Optional[str] = Field(None, description="视频封面URL")

class BatchDownloadRequest(BaseDownloadParams):
    """批量下载请求"""
    videos: List[VideoInfo] = Field(..., description="要下载的视频列表")

async def stream_process_output(process: subprocess.Popen):
    """实时流式输出进程的输出"""
    try:
        # 创建异步迭代器来读取输出
        async def read_output():
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, process.stdout.readline)
                if not line:
                    break
                yield line.strip()

        # 实时读取并发送标准输出
        async for line in read_output():
            if line:
                yield f"data: {line}\n\n"
                # 立即刷新输出
                await asyncio.sleep(0)

        # 等待进程完成
        return_code = await asyncio.get_event_loop().run_in_executor(None, process.wait)

        # 读取可能的错误输出
        stderr_output = await asyncio.get_event_loop().run_in_executor(None, process.stderr.read)
        if stderr_output:
            # 将错误输出按行分割并发送
            for line in stderr_output.strip().split('\n'):
                yield f"data: ERROR: {line}\n\n"
                await asyncio.sleep(0)

        # 发送完成事件
        if return_code == 0:
            yield "data: 下载完成\n\n"
        else:
            # 如果有错误码，发送更详细的错误信息
            yield f"data: 下载失败，错误码：{return_code}\n\n"
            # 尝试获取更多错误信息
            try:
                if process.stderr:
                    process.stderr.seek(0)
                    full_error = process.stderr.read()
                    if full_error:
                        yield f"data: 完整错误信息:\n{full_error}\n\n"
            except Exception as e:
                yield f"data: 无法获取完整错误信息：{str(e)}\n\n"

    except Exception as e:
        yield f"data: 处理过程出错：{str(e)}\n\n"
        import traceback
        yield f"data: 错误堆栈:\n{traceback.format_exc()}\n\n"
    finally:
        # 确保进程已结束
        if process.poll() is None:
            process.terminate()
            await asyncio.get_event_loop().run_in_executor(None, process.wait)
        yield "event: close\ndata: close\n\n"

@router.post("/download_video", summary="下载 B 站视频")
async def download_video(request: DownloadRequest):
    """
    下载 B 站视频

    Args:
        request: 包含视频 URL 和可选 SESSDATA 的请求对象
    """
    try:
        # 检查下载目录和临时目录
        download_dir, tmp_dir = check_download_directories()

        # 构建基本命令
        command = [
            request.url,
            '--dir', download_dir,
            '--tmp-dir', tmp_dir,
            '--subpath-template', f'{{title}}_{{username}}_{{download_date@%Y%m%d_%H%M%S}}_{request.cid}/{{title}}_{request.cid}',
            '--with-metadata'  # 添加元数据文件保存
        ]

        # 添加下载参数
        command = add_download_params_to_command(command, request)
        
        print(f"执行下载命令：yutto {' '.join(command)}")
        
        async def event_stream():
            async for chunk in run_yutto(command):
                yield chunk
            yield "event: close\ndata: close\n\n"
            
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="找不到 yutto 命令，请确保已正确安装"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"下载过程出错：{str(e)}"
        )

@router.get("/check_ffmpeg", summary="检查 FFmpeg 版本")
async def check_ffmpeg():
    """
    检查 FFmpeg 是否安装及其版本信息

    Returns:
        如果安装了 FFmpeg，返回版本信息
        如果未安装，返回简单的未安装信息
    """
    try:
        # 获取系统信息
        import platform
        system = platform.system().lower()
        release = platform.release()
        os_info = {
            "system": system,
            "release": release,
            "platform": platform.platform()
        }

        # 根据不同系统使用不同的命令检查 FFmpeg
        if system == 'windows':
            ffmpeg_check_cmd = 'where ffmpeg'
        else:
            ffmpeg_check_cmd = 'which ffmpeg'

        # 检查 FFmpeg 是否安装
        ffmpeg_process = subprocess.run(
            ffmpeg_check_cmd.split(),
            capture_output=True,
            text=True
        )

        if ffmpeg_process.returncode == 0:
            # FFmpeg 已安装，获取版本信息
            version_process = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if version_process.returncode == 0:
                version_info = version_process.stdout.splitlines()[0]
                return {
                    "status": "success",
                    "installed": True,
                    "version": version_info,
                    "path": ffmpeg_process.stdout.strip(),
                    "os_info": os_info
                }

        # FFmpeg 未安装，返回简单的未安装信息
        return {
            "status": "error",
            "installed": False,
            "message": "FFmpeg 未安装",
            "os_info": os_info
        }

    except Exception as e:
        return {
            "status": "error",
            "installed": False,
            "message": f"检查 FFmpeg 失败：{str(e)}",
            "error": str(e),
            "os_info": os_info if 'os_info' in locals() else {
                "system": platform.system().lower(),
                "release": platform.release(),
                "platform": platform.platform()
            }
        }

@router.get("/check_video_download", summary="检查视频是否已下载")
async def check_video_download(cids: str):
    """
    检查指定 CID 的视频是否已下载，如果已下载则返回保存路径
    支持批量检查多个 CID，使用逗号分隔

    Args:
        cids: 视频的 CID，多个 CID 用逗号分隔，如"12345,67890"

    Returns:
        dict: 包含检查结果和视频保存信息的字典
    """
    try:
        # 解析 CID 列表
        cid_list = [int(cid.strip()) for cid in cids.split(",") if cid.strip()]

        if not cid_list:
            return {
                "status": "error",
                "message": "未提供有效的 CID"
            }

        # 获取下载目录路径
        download_dir = os.path.normpath(config['yutto']['basic']['dir'])

        # 确保下载目录存在
        if not os.path.exists(download_dir):
            return {
                "status": "success",
                "results": {cid: {"downloaded": False, "message": "下载目录不存在，视频尚未下载"} for cid in cid_list}
            }

        # 存储每个 CID 的检查结果
        result_dict = {}

        # 递归遍历下载目录查找匹配的视频文件
        for cid in cid_list:
            found_files = []
            found_directory = None
            download_time = None

            for root, dirs, files in os.walk(download_dir):
                # 检查目录名是否包含 CID
                dir_name = os.path.basename(root)
                if f"_{cid}" in dir_name:
                    found_directory = root

                    # 从目录名中提取下载时间
                    try:
                        # 首先尝试从目录名中直接提取
                        download_time = extract_datetime_from_string(dir_name)

                        # 如果没找到，尝试从目录名的各个部分提取
                        if not download_time:
                            dir_parts = dir_name.split('_')
                            for part in dir_parts:
                                extracted_time = extract_datetime_from_string(part)
                                if extracted_time:
                                    download_time = extracted_time
                                    break

                        # 如果仍然没找到，尝试使用文件的创建时间
                        if not download_time and files:  # 确保有文件存在
                            # 使用第一个文件的创建时间
                            first_file_path = os.path.join(root, files[0])
                            if os.path.exists(first_file_path):
                                creation_time = os.path.getctime(first_file_path)
                                download_time = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
                                print(f"【调试】使用文件创建时间作为下载时间：{download_time}")

                        # 额外记录调试信息
                        if not download_time:
                            print(f"【调试】无法从目录名提取日期时间：{dir_name}")
                            print(f"【调试】目录名各部分：{dir_name.split('_')}")
                    except Exception as e:
                        print(f"提取下载时间出错：{str(e)}")

                    # 检查目录中的文件
                    for file in files:
                        # 检查文件名是否包含 CID
                        if f"_{cid}" in file:
                            # 检查是否为视频或音频文件
                            if file.endswith(('.mp4', '.flv', '.m4a', '.mp3')):
                                file_path = os.path.join(root, file)
                                file_size = os.path.getsize(file_path)
                                file_size_mb = round(file_size / (1024 * 1024), 2)

                                found_files.append({
                                    "file_name": file,
                                    "file_path": file_path,
                                    "size_bytes": file_size,
                                    "size_mb": file_size_mb,
                                    "created_time": os.path.getctime(file_path),
                                    "modified_time": os.path.getmtime(file_path)
                                })

            if found_files:
                result_dict[cid] = {
                    "downloaded": True,
                    "message": f"已找到{len(found_files)}个匹配的视频文件",
                    "files": found_files,
                    "directory": found_directory,
                    "download_time": download_time
                }
            else:
                result_dict[cid] = {
                    "downloaded": False,
                    "message": "未找到已下载的视频文件"
                }

        return {
            "status": "success",
            "results": result_dict
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"检查视频下载状态时出错：{str(e)}"
        }

@router.get("/list_downloaded_videos", summary="获取或搜索已下载视频列表")
async def list_downloaded_videos(search_term: Optional[str] = None, limit: int = 100, page: int = 1, use_local_images: bool = False):
    """
    获取已下载的视频列表，支持通过标题搜索

    Args:
        search_term: 可选，搜索关键词，会在文件名和目录名中查找
        limit: 每页返回的结果数量，默认 100
        page: 页码，从 1 开始，默认为第 1 页
        use_local_images: 是否使用本地图片，默认为 false

    Returns:
        dict: 包含已下载视频列表的字典
    """
    try:
        # 获取下载目录路径
        download_dir = os.path.normpath(config['yutto']['basic']['dir'])

        # 确保下载目录存在
        if not os.path.exists(download_dir):
            return {
                "status": "success",
                "message": "下载目录不存在，尚未下载任何视频",
                "videos": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0
            }

        # 获取数据库连接
        try:
            import sqlite3
            db_path = os.path.join('output', 'bilibili_history.db')
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # 将结果转换为字典形式
            db_available = True
        except Exception as e:
            print(f"无法连接到数据库：{str(e)}")
            db_available = False
            conn = None

        # 递归遍历下载目录查找视频文件
        videos = []

        for root, dirs, files in os.walk(download_dir):
            # 过滤仅包含视频文件的目录
            video_files = []
            dir_name = os.path.basename(root)

            # 如果指定了搜索关键词，检查目录名
            if search_term and search_term.lower() not in dir_name.lower():
                # 跳过不匹配的目录，除非发现其中的文件名匹配
                file_match = False
                for file in files:
                    if search_term.lower() in file.lower() and file.endswith(('.mp4', '.flv', '.m4a', '.mp3')):
                        file_match = True
                        break

                if not file_match:
                    continue

            # 检查是否存在元数据文件
            metadata_file = os.path.join(root, "metadata.json")
            metadata = None
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    print(f"【调试】从元数据文件获取数据：{metadata_file}")

                    # 显示元数据文件内容摘要
                    if 'title' in metadata:
                        print(f"【调试】元数据标题：{metadata['title']}")
                    if 'id' in metadata:
                        if 'bvid' in metadata['id']:
                            print(f"【调试】元数据 BVID: {metadata['id']['bvid']}")
                        if 'cid' in metadata['id']:
                            print(f"【调试】元数据 CID: {metadata['id']['cid']}")
                    if 'owner' in metadata and 'name' in metadata['owner']:
                        print(f"【调试】元数据作者：{metadata['owner']['name']}")
                    if 'cover_url' in metadata:
                        print(f"【调试】元数据封面：{metadata['cover_url']}")

                except Exception as e:
                    print(f"读取元数据文件出错：{str(e)}")

            # 尝试查找.nfo 文件
            nfo_files = [f for f in files if f.endswith('.nfo')]
            nfo_data = None
            if nfo_files:
                try:
                    import xml.etree.ElementTree as ET
                    nfo_file = os.path.join(root, nfo_files[0])
                    tree = ET.parse(nfo_file)
                    nfo_data = tree.getroot()
                    print(f"【调试】从 NFO 文件获取数据：{nfo_file}")
                except Exception as e:
                    print(f"读取 NFO 文件出错：{str(e)}")

            for file in files:
                # 检查是否为视频或音频文件
                if file.endswith(('.mp4', '.flv', '.m4a', '.mp3')):
                    # 如果指定了搜索关键词，检查文件名
                    if search_term and search_term.lower() not in file.lower() and search_term.lower() not in dir_name.lower():
                        continue

                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_size_mb = round(file_size / (1024 * 1024), 2)

                    # 从目录名和文件名中提取信息
                    dir_parts = dir_name.split('_')
                    file_parts = file.split('_')

                    # 尝试提取 CID
                    cid = None
                    try:
                        if len(dir_parts) > 3:
                            cid = dir_parts[-1]  # 最后一部分应该是 CID
                        elif len(file_parts) > 1:
                            cid = file_parts[-1].split('.')[0]  # 文件名最后一部分的。前部分
                    except:
                        pass

                    # 尝试从目录名提取标题
                    title = None
                    try:
                        if len(dir_parts) > 0:
                            # 除去最后 3 个部分（用户名_日期_CID），剩下的应该是标题
                            title = '_'.join(dir_parts[:-3]) if len(dir_parts) > 3 else dir_name
                    except:
                        title = dir_name

                    # 尝试提取日期时间
                    date_time = None
                    try:
                        print(f"【调试】处理目录：{dir_name}")

                        # 首先尝试从完整目录名中直接提取
                        date_time = extract_datetime_from_string(dir_name)

                        # 如果没找到，尝试从目录名的各个部分提取
                        if not date_time:
                            dir_parts = dir_name.split('_')
                            print(f"【调试】目录名各部分：{dir_parts}")
                            for part in dir_parts:
                                print(f"【调试】检查部分：{part}")
                                extracted_time = extract_datetime_from_string(part)
                                if extracted_time:
                                    date_time = extracted_time
                                    print(f"【调试】从部分'{part}'提取到时间：{date_time}")
                                    break

                        # 如果仍然没找到，尝试使用文件的创建时间
                        if not date_time:
                            # 使用即将添加到 video_files 的文件创建时间
                            creation_time = os.path.getctime(file_path)
                            date_time = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
                            print(f"【调试】使用文件创建时间作为下载时间：{date_time}")

                            # 额外记录调试信息
                            print(f"【调试】无法从目录名提取日期时间：{dir_name}")
                    except Exception as e:
                        print(f"提取下载时间出错：{str(e)}")

                    video_files.append({
                        "file_name": file,
                        "file_path": file_path,
                        "size_bytes": file_size,
                        "size_mb": file_size_mb,
                        "created_time": os.path.getctime(file_path),
                        "modified_time": os.path.getmtime(file_path),
                        "is_audio_only": file.endswith(('.m4a', '.mp3'))
                    })

            if video_files:
                video_info = {
                    "directory": root,
                    "dir_name": dir_name,
                    "title": title,
                    "cid": cid,
                    "bvid": None,  # 初始化 bvid 字段为 None
                    "download_date": date_time,
                    "files": video_files,
                    "cover": None,
                    "author_face": None,
                    "author_name": None,
                    "author_mid": None
                }

                # 如果存在元数据，优先使用元数据中的信息
                if metadata:
                    try:
                        # 提取 bvid 和 cid
                        if 'id' in metadata:
                            video_id = metadata['id']
                            if 'bvid' in video_id:
                                video_info["bvid"] = video_id['bvid']
                            if 'cid' in video_id and not video_info["cid"]:
                                video_info["cid"] = str(video_id['cid'])

                        # 提取标题
                        if 'title' in metadata and metadata['title']:
                            video_info["title"] = metadata['title']

                        # 提取封面 URL
                        if 'cover_url' in metadata and metadata['cover_url']:
                            video_info["cover"] = metadata['cover_url']

                        # 提取作者信息
                        if 'owner' in metadata:
                            owner = metadata['owner']
                            if 'name' in owner:
                                video_info["author_name"] = owner['name']
                            if 'face' in owner:
                                video_info["author_face"] = owner['face']
                            if 'mid' in owner:
                                video_info["author_mid"] = owner['mid']

                        # 处理图片 URL
                        if _process_image_url:
                            # 使用导入的函数处理图片 URL
                            if video_info["cover"]:
                                video_info["cover"] = _process_image_url(video_info["cover"], 'covers', use_local_images)
                            if video_info["author_face"]:
                                video_info["author_face"] = _process_image_url(video_info["author_face"], 'avatars', use_local_images)
                        elif hasattr(sys.modules.get('routers.history'), '_process_image_url'):
                            # 如果导入失败但模块运行时可访问，再次尝试
                            process_url = getattr(sys.modules.get('routers.history'), '_process_image_url')
                            if video_info["cover"]:
                                video_info["cover"] = process_url(video_info["cover"], 'covers', use_local_images)
                            if video_info["author_face"]:
                                video_info["author_face"] = process_url(video_info["author_face"], 'avatars', use_local_images)
                        elif use_local_images:
                            # 简单的 URL 处理逻辑，作为后备方案
                            import hashlib
                            if video_info["cover"]:
                                cover_hash = hashlib.md5(video_info["cover"].encode()).hexdigest()
                                video_info["cover"] = f"http://localhost:8899/images/local/covers/{cover_hash}"
                            if video_info["author_face"]:
                                avatar_hash = hashlib.md5(video_info["author_face"].encode()).hexdigest()
                                video_info["author_face"] = f"http://localhost:8899/images/local/avatars/{avatar_hash}"

                        print(f"【调试】从元数据获取到视频信息：{video_info['title']}，封面 URL: {video_info['cover'][:50]}...")
                    except Exception as e:
                        print(f"解析元数据时出错：{str(e)}")

                # 如果有 NFO 数据且信息不完整，尝试从 NFO 提取
                if nfo_data and (not video_info["cover"] or not video_info["author_name"] or not video_info["author_face"]):
                    try:
                        # 提取标题
                        title_elem = nfo_data.find('title')
                        if title_elem is not None and title_elem.text and not video_info["title"]:
                            video_info["title"] = title_elem.text

                        # 提取封面 URL
                        thumb_elem = nfo_data.find('thumb')
                        if thumb_elem is not None and thumb_elem.text and not video_info["cover"]:
                            video_info["cover"] = thumb_elem.text

                        # 提取作者信息
                        actor_elem = nfo_data.find('actor')
                        if actor_elem is not None:
                            # 作者名
                            actor_name = actor_elem.find('name')
                            if actor_name is not None and actor_name.text and not video_info["author_name"]:
                                video_info["author_name"] = actor_name.text

                            # 作者头像
                            actor_thumb = actor_elem.find('thumb')
                            if actor_thumb is not None and actor_thumb.text and not video_info["author_face"]:
                                video_info["author_face"] = actor_thumb.text

                            # 作者 ID/主页
                            actor_profile = actor_elem.find('profile')
                            if actor_profile is not None and actor_profile.text and not video_info["author_mid"]:
                                profile_url = actor_profile.text
                                # 尝试从 URL 中提取 mid
                                mid_match = re.search(r"space\.bilibili\.com/(\d+)", profile_url)
                                if mid_match:
                                    video_info["author_mid"] = int(mid_match.group(1))

                        # 提取 BV 号
                        website_elem = nfo_data.find('website')
                        if website_elem is not None and website_elem.text and not video_info["bvid"]:
                            bvid_match = re.search(r"video/(BV\w+)", website_elem.text)
                            if bvid_match:
                                video_info["bvid"] = bvid_match.group(1)

                        # 处理 NFO 文件中的图片 URL
                        if _process_image_url:
                            # 使用导入的函数处理图片 URL
                            if video_info["cover"]:
                                video_info["cover"] = _process_image_url(video_info["cover"], 'covers', use_local_images)
                            if video_info["author_face"]:
                                video_info["author_face"] = _process_image_url(video_info["author_face"], 'avatars', use_local_images)
                        elif hasattr(sys.modules.get('routers.history'), '_process_image_url'):
                            # 如果导入失败但模块运行时可访问，再次尝试
                            process_url = getattr(sys.modules.get('routers.history'), '_process_image_url')
                            if video_info["cover"]:
                                video_info["cover"] = process_url(video_info["cover"], 'covers', use_local_images)
                            if video_info["author_face"]:
                                video_info["author_face"] = process_url(video_info["author_face"], 'avatars', use_local_images)
                        elif use_local_images:
                            # 简单的 URL 处理逻辑，作为后备方案
                            import hashlib
                            if video_info["cover"]:
                                cover_hash = hashlib.md5(video_info["cover"].encode()).hexdigest()
                                video_info["cover"] = f"http://localhost:8899/images/local/covers/{cover_hash}"
                            if video_info["author_face"]:
                                avatar_hash = hashlib.md5(video_info["author_face"].encode()).hexdigest()
                                video_info["author_face"] = f"http://localhost:8899/images/local/avatars/{avatar_hash}"

                        print(f"【调试】从 NFO 文件获取到视频信息：{video_info['title']}，封面 URL: {video_info['cover'][:50] if video_info['cover'] else 'None'}")
                    except Exception as e:
                        print(f"解析 NFO 文件时出错：{str(e)}")

                # 如果有 CID 但没有其他信息，尝试通过 API 获取
                if not metadata and not nfo_data and cid and cid.isdigit() and (not video_info["cover"] or not video_info["author_name"] or not video_info["author_face"]):
                    try:
                        # 仅当没有元数据和 NFO 文件时，才尝试通过 API 或数据库获取
                        print(f"【调试】没有找到元数据或 NFO 文件，尝试通过 API/数据库获取 CID={cid}的视频信息")

                        # 方式 1: 直接调用 get_video_by_cid 函数（如果已成功导入）
                        if get_video_by_cid:
                            print(f"【调试】使用导入的 get_video_by_cid 函数获取 CID={cid}的视频信息")
                            # 调用 API 函数获取视频信息
                            api_response = await get_video_by_cid(int(cid), use_local_images)

                            if api_response["status"] == "success" and "data" in api_response:
                                video_data = api_response["data"]
                                video_info["title"] = video_data.get("title") or video_info["title"]
                                video_info["cover"] = video_data.get("cover")
                                video_info["author_face"] = video_data.get("author_face")
                                video_info["author_name"] = video_data.get("author_name")
                                video_info["author_mid"] = video_data.get("author_mid")
                                video_info["bvid"] = video_data.get("bvid")  # 添加 bvid 字段
                                print(f"【调试】成功通过 API 获取到视频信息：{video_data.get('title')}")
                        # 方式 2: 如果 API 函数未导入，则回退到直接查询数据库
                        elif db_available:
                            print(f"【调试】回退到直接查询数据库获取 CID={cid}的视频信息")
                            cursor = conn.cursor()
                            # 查询所有历史记录表
                            years = [table_name.split('_')[-1]
                                    for (table_name,) in cursor.execute(
                                        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'bilibili_history_%'"
                                    ).fetchall()
                                    if table_name.split('_')[-1].isdigit()]

                            # 构建 UNION ALL 查询所有年份表
                            if years:
                                queries = []
                                for year in years:
                                    queries.append(f"SELECT title, cover, author_face, author_name, author_mid, bvid FROM bilibili_history_{year} WHERE cid = {cid} LIMIT 1")

                                # 执行联合查询
                                union_query = " UNION ALL ".join(queries) + " LIMIT 1"
                                result = cursor.execute(union_query).fetchone()

                                if result:
                                    # 设置封面和作者信息
                                    video_info["title"] = result["title"] or video_info["title"]
                                    video_info["cover"] = result["cover"]
                                    video_info["author_face"] = result["author_face"]
                                    video_info["author_name"] = result["author_name"]
                                    video_info["author_mid"] = result["author_mid"]
                                    video_info["bvid"] = result["bvid"]  # 添加 bvid 字段

                                    # 处理图片 URL
                                    if _process_image_url:
                                        # 使用导入的函数处理图片 URL
                                        if video_info["cover"]:
                                            video_info["cover"] = _process_image_url(video_info["cover"], 'covers', use_local_images)
                                        if video_info["author_face"]:
                                            video_info["author_face"] = _process_image_url(video_info["author_face"], 'avatars', use_local_images)
                                    elif hasattr(sys.modules.get('routers.history'), '_process_image_url'):
                                        # 如果导入失败但模块运行时可访问，再次尝试
                                        process_url = getattr(sys.modules.get('routers.history'), '_process_image_url')
                                        if video_info["cover"]:
                                            video_info["cover"] = process_url(video_info["cover"], 'covers', use_local_images)
                                        if video_info["author_face"]:
                                            video_info["author_face"] = process_url(video_info["author_face"], 'avatars', use_local_images)
                                    else:
                                        # 简单的 URL 处理逻辑，作为后备方案
                                        if use_local_images:
                                            import hashlib
                                            if video_info["cover"]:
                                                cover_hash = hashlib.md5(video_info["cover"].encode()).hexdigest()
                                                video_info["cover"] = f"http://localhost:8899/images/local/covers/{cover_hash}"
                                            if video_info["author_face"]:
                                                avatar_hash = hashlib.md5(video_info["author_face"].encode()).hexdigest()
                                                video_info["author_face"] = f"http://localhost:8899/images/local/avatars/{avatar_hash}"
                    except Exception as e:
                        print(f"获取视频信息时出错：{str(e)}")

                videos.append(video_info)

        # 如果数据库连接已打开，关闭它
        if conn:
            conn.close()

        # 计算分页
        total_videos = len(videos)
        total_pages = (total_videos + limit - 1) // limit if total_videos > 0 else 0

        # 根据修改时间排序，最新的在前面
        videos.sort(key=lambda x: max([f["modified_time"] for f in x["files"]]) if x["files"] else 0, reverse=True)

        # 分页
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total_videos)
        paginated_videos = videos[start_idx:end_idx] if start_idx < total_videos else []

        return {
            "status": "success",
            "message": f"找到{total_videos}个视频" + (f"，匹配'{search_term}'" if search_term else ""),
            "videos": paginated_videos,
            "total": total_videos,
            "page": page,
            "limit": limit,
            "pages": total_pages
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"获取已下载视频列表时出错：{str(e)}"
        }

@router.get("/stream_video", summary="获取已下载视频的流媒体数据")
async def stream_video(file_path: str):
    """
    返回已下载视频的流媒体数据，用于在线播放

    Args:
        file_path: 视频文件的完整路径

    Returns:
        StreamingResponse: 视频流响应
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在：{file_path}"
            )

        # 检查是否是支持的媒体文件
        if not file_path.endswith(('.mp4', '.flv', '.m4a', '.mp3')):
            raise HTTPException(
                status_code=400,
                detail="不支持的媒体文件格式，仅支持 mp4、flv、m4a、mp3 格式"
            )

        # 获取文件大小
        file_size = os.path.getsize(file_path)

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 设置适当的媒体类型
        if file_path.endswith('.mp4'):
            media_type = 'video/mp4'
        elif file_path.endswith('.flv'):
            media_type = 'video/x-flv'
        elif file_path.endswith('.m4a'):
            media_type = 'audio/mp4'
        elif file_path.endswith('.mp3'):
            media_type = 'audio/mpeg'
        else:
            media_type = 'application/octet-stream'

        # 返回文件响应
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=file_name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取视频流时出错：{str(e)}"
        )

@router.delete("/delete_downloaded_video", summary="删除已下载的视频")
async def delete_downloaded_video(
    delete_directory: bool = False,
    directory: Optional[str] = None,
    cid: Optional[int] = Query(None, description="视频的 CID，可选项")
):
    """
    删除已下载的视频文件

    Args:
        delete_directory: 是否删除整个目录，默认为 False（只删除视频文件）
        directory: 可选，指定要删除文件的目录路径，如果提供则只在该目录中查找和删除文件
        cid: 可选，视频的 CID

    Returns:
        dict: 包含删除结果信息的字典
    """
    try:
        # 获取下载目录路径
        download_dir = os.path.normpath(config['yutto']['basic']['dir'])

        # 确保下载目录存在
        if not os.path.exists(download_dir):
            return {
                "status": "error",
                "message": "下载目录不存在"
            }

        # 检查参数有效性
        if not cid and not directory:
            return {
                "status": "error",
                "message": "必须提供 cid 或 directory 参数中的至少一个"
            }

        # 查找匹配 CID 的视频文件和目录
        found_files = []
        found_directory = directory  # 如果提供了目录，则使用它

        # 如果提供了 directory 参数，并且它确实存在，只在该目录中查找文件
        if directory and os.path.exists(directory):
            # 根据 directory 直接处理
            if delete_directory:
                # 删除整个目录
                import shutil
                try:
                    shutil.rmtree(directory)
                    return {
                        "status": "success",
                        "message": f"已删除目录：{directory}",
                        "deleted_directory": directory
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"删除目录时出错：{str(e)}",
                        "directory": directory
                    }
            else:
                # 查找目录中的视频文件并删除
                for file in os.listdir(directory):
                    # 仅查找视频或音频文件
                        if file.endswith(('.mp4', '.flv', '.m4a', '.mp3')):
                            file_path = os.path.join(directory, file)
                        if cid is None or f"_{cid}" in file:  # 如果指定了 CID 则检查文件名是否包含它
                            found_files.append({
                                "file_name": file,
                                "file_path": file_path
                            })
        elif cid is not None:
            # 如果没有提供 directory 参数但提供了 cid，执行原来的逻辑
            for root, dirs, files in os.walk(download_dir):
                # 检查目录名是否包含 CID
                if f"_{cid}" in os.path.basename(root):
                    # 如果没有指定目录，保存找到的第一个匹配目录
                    if not found_directory:
                        found_directory = root

                        # 检查目录中的文件
                        for file in files:
                        # 检查文件名是否包含 CID
                            if f"_{cid}" in file:
                                # 检查是否为视频或音频文件
                                if file.endswith(('.mp4', '.flv', '.m4a', '.mp3')):
                                    file_path = os.path.join(root, file)
                                    found_files.append({
                                        "file_name": file,
                                        "file_path": file_path
                                    })

        if not found_files and not found_directory:
            error_message = "未找到匹配的视频文件"
            if cid is not None:
                error_message += f", CID: {cid}"
            if directory:
                error_message += f"，目录：{directory}"

            return {
                "status": "error",
                "message": error_message
            }

        # 执行删除操作
        deleted_files = []

        if delete_directory and found_directory:
            # 删除整个目录
            import shutil
            try:
                shutil.rmtree(found_directory)
                return {
                    "status": "success",
                    "message": f"已删除目录：{found_directory}",
                    "deleted_directory": found_directory
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"删除目录时出错：{str(e)}",
                    "directory": found_directory
                }
        else:
            # 只删除视频文件
            for file_info in found_files:
                try:
                    os.remove(file_info["file_path"])
                    deleted_files.append(file_info)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"删除文件时出错：{str(e)}",
                        "file": file_info["file_path"]
                    }

            return {
                "status": "success",
                "message": f"已删除{len(deleted_files)}个文件",
                "deleted_files": deleted_files,
                "directory": found_directory
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"删除视频文件时出错：{str(e)}"
        }

@router.get("/stream_danmaku", summary="获取视频弹幕文件")
async def stream_danmaku(file_path: Optional[str] = None, cid: Optional[int] = None):
    """
    返回视频弹幕文件 (.ass)，用于前端播放时显示弹幕

    Args:
        file_path: 视频文件的完整路径，会自动查找对应的 ass 文件
        cid: 可选，如果提供 CID 而不是文件路径，将尝试查找对应 CID 的弹幕文件

    Returns:
        FileResponse: 弹幕文件响应
    """
    try:
        if not file_path and not cid:
            raise HTTPException(
                status_code=400,
                detail="必须提供视频文件路径 (file_path) 或视频 CID(cid) 参数"
            )

        danmaku_path = None

        # 1. 如果提供了文件路径，尝试查找对应的 ass 文件
        if file_path:
            # 检查视频文件是否存在
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"视频文件不存在：{file_path}"
                )

            # 尝试找到同名的.ass 文件
            base_path = file_path.rsplit('.', 1)[0]  # 移除扩展名
            possible_ass_path = f"{base_path}.ass"

            if os.path.exists(possible_ass_path):
                danmaku_path = possible_ass_path
            else:
                # 尝试在同一目录下查找任何包含相同 CID 的.ass 文件
                directory = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)

                # 尝试从文件名提取 CID
                cid_match = None
                file_parts = file_name.split('_')
                if len(file_parts) > 1:
                    try:
                        # 尝试获取最后一部分中的 CID (去掉扩展名)
                        last_part = file_parts[-1].split('.')[0]
                        if last_part.isdigit():
                            cid_match = last_part
                    except:
                        pass

                if cid_match:
                    # 在同一目录下查找包含相同 CID 的.ass 文件
                    for file in os.listdir(directory):
                        if file.endswith('.ass') and cid_match in file:
                            danmaku_path = os.path.join(directory, file)
                            break

        # 2. 如果提供了 CID，在下载目录中查找对应的弹幕文件
        elif cid:
            download_dir = os.path.normpath(config['yutto']['basic']['dir'])

            # 确保下载目录存在
            if not os.path.exists(download_dir):
                raise HTTPException(
                    status_code=404,
                    detail="下载目录不存在"
                )

            # 递归遍历下载目录查找匹配 CID 的弹幕文件
            for root, dirs, files in os.walk(download_dir):
                # 检查目录名是否包含 CID
                if f"_{cid}" in os.path.basename(root):
                    # 检查目录中的文件
                    for file in files:
                        # 检查是否为弹幕文件
                        if file.endswith('.ass') and f"_{cid}" in file:
                            danmaku_path = os.path.join(root, file)
                            break

                    # 如果在当前目录找到了弹幕文件，就不再继续查找
                    if danmaku_path:
                        break

        # 检查是否找到弹幕文件
        if not danmaku_path:
            raise HTTPException(
                status_code=404,
                detail=f"未找到匹配的弹幕文件"
            )

        # 返回文件响应
        return FileResponse(
            danmaku_path,
            media_type='text/plain',
            filename=os.path.basename(danmaku_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取弹幕文件时出错：{str(e)}"
        )

@router.post("/download_user_videos", summary="下载用户全部投稿视频")
async def download_user_videos(request: UserSpaceDownloadRequest):
    """
    下载指定用户的全部投稿视频

    Args:
        request: 包含用户 ID 和可选 SESSDATA 的请求对象
    """
    try:
        # 检查下载目录和临时目录
        download_dir, tmp_dir = check_download_directories()

        # 构建用户空间 URL
        user_space_url = f"https://space.bilibili.com/{request.user_id}/video"

        # 构建基本命令
        command = [
            user_space_url,
            '--batch',  # 批量下载
            '--dir', download_dir,
            '--tmp-dir', tmp_dir,
            '--subpath-template', f'{{username}}的全部投稿视频/{{title}}_{{download_date@%Y%m%d_%H%M%S}}/{{title}}',
            '--with-metadata'  # 添加元数据文件保存
        ]

        # 添加下载参数
        command = add_download_params_to_command(command, request)
        
        print(f"执行下载命令：yutto {' '.join(command)}")
        
        async def event_stream():
            async for chunk in run_yutto(command):
                yield chunk
            yield "event: close\ndata: close\n\n"
            
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载过程出错：{str(e)}")

@router.post("/batch_download", summary="批量下载多个B站视频")
async def batch_download(request: BatchDownloadRequest):
    """
    批量下载多个B站视频

    Args:
        request: 包含多个视频信息和下载选项的请求对象
    """
    try:
        # 检查下载目录和临时目录
        download_dir, tmp_dir = check_download_directories()

        # 准备批量下载
        total_videos = len(request.videos)
        current_index = 0

        # 创建一个异步生成器来处理批量下载
        async def batch_download_generator():
            nonlocal current_index

            # 发送初始信息
            yield f"data: 开始批量下载，共 {total_videos} 个视频\n\n"

            for video in request.videos:
                current_index += 1

                # 发送当前下载信息
                yield f"data: 正在下载第 {current_index}/{total_videos} 个视频: {video.title or video.bvid}\n\n"

                # 构建视频 URL
                video_url = f"https://www.bilibili.com/video/{video.bvid}"

                # -------------------- 组装 yutto 参数 --------------------
                argv = [
                    video_url,
                    '--dir', download_dir,
                    '--tmp-dir', tmp_dir,
                    '--subpath-template',
                    f'{{title}}_{{username}}_{{download_date@%Y%m%d_%H%M%S}}_{video.cid}/{{title}}_{video.cid}',
                    '--with-metadata'  # 保存元数据文件
                ]
                # 注：add_download_params_to_command 内部会按需追加其它参数
                argv = add_download_params_to_command(argv, request)

                # 打印调试信息
                print("执行下载命令：yutto " + ' '.join(argv))

                # -------------------- 执行下载 --------------------
                try:
                    # 进程内调用 yutto，并实时转发输出
                    async for line in run_yutto(argv):
                        yield line

                    # 发送当前视频下载完成信息
                    yield f"data: 第 {current_index}/{total_videos} 个视频下载完成: {video.title or video.bvid}\n\n"

                except Exception as e:
                    error_msg = f"下载视频 {video.bvid} 时出错：{str(e)}"
                    print(error_msg)
                    yield f"data: ERROR: {error_msg}\n\n"

            # 发送批量下载完成信息
            yield f"data: 批量下载完成，共 {total_videos} 个视频\n\n"
            yield "event: close\ndata: close\n\n"

        # 返回 SSE 响应
        return StreamingResponse(
            batch_download_generator(),
            media_type="text/event-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量下载过程出错：{str(e)}")

@router.post("/download_favorites", summary="下载用户收藏夹视频")
async def download_favorites(request: FavoriteDownloadRequest):
    """
    下载用户的收藏夹视频

    Args:
        request: 包含用户 ID、收藏夹 ID 和可选 SESSDATA 的请求对象
        注意：不提供收藏夹 ID 时，将下载所有收藏夹
    """
    try:
        # 收藏夹必须登录
        sessdata = request.sessdata or config.get('SESSDATA')
        if not sessdata:
            raise HTTPException(
                status_code=401,
                detail="未登录：下载收藏夹必须提供 SESSDATA"
            )

        # 检查下载目录和临时目录
        download_dir, tmp_dir = check_download_directories()

        # 构建收藏夹 URL
        if request.fid:
            # 指定收藏夹
            favorite_url = f"https://space.bilibili.com/{request.user_id}/favlist?fid={request.fid}"
        else:
            # 所有收藏夹
            favorite_url = f"https://space.bilibili.com/{request.user_id}/favlist"

        # 构建基本命令
        command = [
            favorite_url,
            '--batch',  # 批量下载
            '--dir', download_dir,
            '--tmp-dir', tmp_dir,
            '--subpath-template', f'{{username}}的收藏夹/{{title}}_{{download_date@%Y%m%d_%H%M%S}}/{{title}}',
            '--with-metadata'  # 添加元数据文件保存
        ]

        # 添加下载参数
        command = add_download_params_to_command(command, request)

        # 确保添加 SESSDATA (收藏夹必须登录)
        if '--sessdata' not in command:
            command.extend(['--sessdata', sessdata])

        print(f"执行下载命令：yutto {' '.join(command)}")
        
        async def event_stream():
            async for chunk in run_yutto(command):
                yield chunk
            yield "event: close\ndata: close\n\n"
            
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载过程出错：{str(e)}")

# 定义响应模型
class VideoInfo(BaseModel):
    path: str
    size: int
    title: str
    create_time: datetime
    cover: str = ""
    cid: str = ""
    bvid: str = ""
    author_name: str = ""
    author_face: str = ""
    author_mid: int = 0

class VideoSearchResponse(BaseModel):
    total: int
    videos: list[VideoInfo]

# 视频详细信息响应模型
class VideoDetailResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

@router.get("/video_info", summary="获取 B 站视频详细信息")
async def get_video_info(aid: Optional[int] = None, bvid: Optional[str] = None, sessdata: Optional[str] = None, headers: Optional[dict] = None, use_sessdata: bool = True):
    """
    获取B站视频详细信息

    Args:
        aid: 视频aid
        bvid: 视频bvid
        sessdata: B站会话ID
        headers: 自定义请求头
        use_sessdata: 是否使用SESSDATA认证，默认为True
    """
    try:
        if not aid and not bvid:
            return VideoDetailResponse(
                status="error",
                message="至少需要提供aid或bvid参数",
                data=None
            )

        # 配置请求信息
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com',
        }

        # 合并自定义请求头
        if headers:
            default_headers.update(headers)

        # 仅在需要使用SESSDATA且提供了SESSDATA时，加入到请求头中
        if sessdata and use_sessdata:
            default_headers['Cookie'] = f'SESSDATA={sessdata};'

        # 准备请求参数
        params = {}
        if aid:
            params['aid'] = aid
        if bvid:
            params['bvid'] = bvid

        url = "https://api.bilibili.com/x/web-interface/view"

        # 使用httpx发送请求
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=default_headers, timeout=20.0)

            # 首先检查内容类型
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type:
                return VideoDetailResponse(
                    status="error",
                    message=f"非JSON响应: {content_type}，视频可能无法访问",
                    data=None
                )

            # 尝试多种解码方式
            content = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin1']:
                try:
                    content = response.content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            # 如果所有解码方式都失败，使用bytes的十六进制表示
            if content is None:
                hex_content = response.content.hex()
                return VideoDetailResponse(
                    status="error",
                    message=f"无法解码响应内容，可能是非文本数据",
                    data={"raw_hex": hex_content[:100] + "..."}
                )

            # 尝试解析JSON
            try:
                response_json = json.loads(content)
            except json.JSONDecodeError:
                return VideoDetailResponse(
                    status="error",
                    message=f"无法解析JSON: {content[:200]}...",
                    data=None
                )

            # 检查是否API错误
            code = response_json.get('code', 0)
            if code != 0:
                error_msg = response_json.get('message', '未知错误')

                # 特殊处理一些常见错误
                if code == -404:
                    error_msg = "视频不存在或已被删除"
                elif code == 62002:
                    error_msg = "视频不可见（可能是私有或被删除）"

                return VideoDetailResponse(
                    status="error",
                    message=f"API错误 {code}: {error_msg}",
                    data=response_json
                )

            # 正常返回
            return VideoDetailResponse(
                status="success",
                message="获取视频信息成功",
                data=response_json.get('data', {})
            )

    except httpx.RequestError as e:
        return VideoDetailResponse(
            status="error",
            message=f"请求错误: {str(e)}",
            data=None
        )
    except Exception as e:
        return VideoDetailResponse(
            status="error",
            message=f"获取视频信息时出错：{str(e)}",
            data=None
        )

# 用户投稿视频查询参数模型
class UserVideosQueryParams(BaseModel):
    mid: int = Field(..., description="目标用户 mid")
    pn: int = Field(1, description="页码")
    ps: int = Field(30, description="每页项数")
    tid: int = Field(0, description="分区筛选，0：全部")
    keyword: str = Field("", description="关键词筛选")
    order: str = Field("pubdate", description="排序方式")
    platform: str = Field("web", description="平台标识")

# 用户投稿视频响应模型
class UserVideosResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

@router.get("/user_videos", summary="查询用户投稿视频明细")
async def get_user_videos(
    mid: int,
    pn: int = 1,
    ps: int = 30,
    tid: int = 0,
    keyword: str = "",
    order: str = "pubdate",
    sessdata: Optional[str] = None,
    use_sessdata: bool = True
):
    """
    查询用户投稿视频明细

    Args:
        mid: 目标用户 mid
        pn: 页码，默认为 1
        ps: 每页项数，默认为 30
        tid: 分区 ID，默认为 0（全部）
        keyword: 关键词过滤，默认为空
        order: 排序方式，默认为 pubdate（发布日期）
               可选值：pubdate（发布日期）、click（播放量）、stow（收藏量）
        sessdata: 可选，用户的 SESSDATA，用于获取限制查看的视频
        use_sessdata: 是否使用SESSDATA认证，默认为True

    Returns:
        用户投稿视频列表
    """
    try:
        # 设置请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Referer': f'https://space.bilibili.com/{mid}/video',
            'Origin': 'https://space.bilibili.com'
        }

        # 仅在需要使用SESSDATA且提供了SESSDATA时，加入到请求头中
        if sessdata and use_sessdata:
            headers['Cookie'] = f'SESSDATA={sessdata}'
        elif config.get('SESSDATA') and use_sessdata:
            headers['Cookie'] = f'SESSDATA={config["SESSDATA"]}'

        # 构建请求参数
        params = {
            'mid': mid,
            'pn': pn,
            'ps': ps,
            'tid': tid,
            'keyword': keyword,
            'order': order,
            'platform': 'web'
        }

        # 使用 WBI 签名
        from scripts.wbi_sign import get_wbi_sign
        signed_params = get_wbi_sign(params)

        # 发送请求获取用户视频列表
        response = requests.get(
            'https://api.bilibili.com/x/space/wbi/arc/search',
            params=signed_params,
            headers=headers,
            timeout=10
        )

        # 显式设置响应编码
        response.encoding = 'utf-8'

        # 打印响应状态和内容预览，便于调试
        print(f"请求 URL: {response.url}")
        print(f"响应状态码：{response.status_code}")
        content_preview = response.text[:100] if len(response.text) > 100 else response.text
        print(f"响应内容预览：{content_preview}")

        # 解析响应
        response_json = response.json()

        # 处理可能的错误
        if response_json.get('code') != 0:
            return UserVideosResponse(
                status="error",
                message=f"获取用户投稿视频列表失败：{response_json.get('message', '未知错误')}",
                data=response_json
            )

        # 返回成功响应
        return UserVideosResponse(
            status="success",
            message="获取用户投稿视频列表成功",
            data=response_json.get('data')
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"获取用户投稿视频列表时出错：{str(e)}")
        print(f"错误堆栈：{error_trace}")

        return UserVideosResponse(
            status="error",
            message=f"获取用户投稿视频列表时出错：{str(e)}",
            data={"error_trace": error_trace}
        )

# 合集视频信息响应模型
class SeasonVideoInfo(BaseModel):
    title: str
    cover: str
    duration: int
    vv: int
    vt: int
    bvid: str
    aid: int
    cid: int

class SeasonInfoResponse(BaseModel):
    status: str
    message: str
    season_id: Optional[int] = None
    season_title: Optional[str] = None
    season_cover: Optional[str] = None
    videos: Optional[List[SeasonVideoInfo]] = None

@router.get("/video_season_info", summary="获取视频观看时长信息")
async def get_video_season_info(bvid: str, sessdata: Optional[str] = None):
    """
    获取视频观看时长信息

    检查视频是否为合集中的视频，并返回合集信息及其中所有视频的详细信息

    Args:
        bvid: 视频bvid
        sessdata: B站会话ID，用于获取需要登录才能访问的视频
    """
    try:
        # 首先调用现有的获取视频详情接口
        video_detail = await get_video_info(bvid=bvid, sessdata=sessdata)

        # 如果获取视频详情失败，直接返回错误
        if video_detail.status != "success":
            return SeasonInfoResponse(
                status="error",
                message=f"获取视频信息失败: {video_detail.message}"
            )

        # 检查视频是否属于某个合集
        video_data = video_detail.data
        if not video_data.get("season_id"):
            return SeasonInfoResponse(
                status="info",
                message="该视频不属于任何合集"
            )

        # 视频属于合集，获取合集信息
        season_id = video_data.get("season_id")

        # 如果有ugc_season字段，从中提取合集信息
        season_info = video_data.get("ugc_season", {})
        if not season_info:
            return SeasonInfoResponse(
                status="error",
                message="无法获取合集信息",
                season_id=season_id
            )

        # 提取合集标题和封面
        season_title = season_info.get("title", "")
        season_cover = season_info.get("cover", "")

        # 提取合集中的所有视频信息
        video_list = []
        sections = season_info.get("sections", [])

        for section in sections:
            episodes = section.get("episodes", [])
            for episode in episodes:
                # 提取所需的视频信息
                arc = episode.get("arc", {})
                page = episode.get("page", {})

                video_info = SeasonVideoInfo(
                    title=episode.get("title", ""),
                    cover=arc.get("pic", ""),
                    duration=page.get("duration", 0),
                    vv=arc.get("stat", {}).get("vv", 0),
                    vt=arc.get("stat", {}).get("vt", 0),
                    bvid=episode.get("bvid", ""),
                    aid=arc.get("aid", 0),
                    cid=page.get("cid", 0)
                )
                video_list.append(video_info)

        return SeasonInfoResponse(
            status="success",
            message="获取合集视频信息成功",
            season_id=season_id,
            season_title=season_title,
            season_cover=season_cover,
            videos=video_list
        )

    except Exception as e:
        return SeasonInfoResponse(
            status="error",
            message=f"获取视频观看时长信息时出错：{str(e)}"
        )