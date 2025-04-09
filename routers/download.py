import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from scripts.utils import load_config

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

class DownloadRequest(BaseModel):
    url: str
    sessdata: Optional[str] = Field(None, description="用户的 SESSDATA")
    download_cover: Optional[bool] = Field(True, description="是否下载视频封面")
    only_audio: Optional[bool] = Field(False, description="是否仅下载音频")
    cid: int = Field(..., description="视频的 CID，用于分类存储和音频文件命名前缀")

class UserSpaceDownloadRequest(BaseModel):
    user_id: str = Field(..., description="用户 UID，例如：100969474")
    sessdata: Optional[str] = Field(None, description="用户的 SESSDATA")
    download_cover: Optional[bool] = Field(True, description="是否下载视频封面")
    only_audio: Optional[bool] = Field(False, description="是否仅下载音频")

class FavoriteDownloadRequest(BaseModel):
    user_id: str = Field(..., description="用户 UID，例如：100969474")
    fid: Optional[str] = Field(None, description="收藏夹 ID，不提供时下载所有收藏夹")
    sessdata: Optional[str] = Field(None, description="用户的 SESSDATA")
    download_cover: Optional[bool] = Field(True, description="是否下载视频封面")
    only_audio: Optional[bool] = Field(False, description="是否仅下载音频")

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
        # 检查登录状态
        if config['yutto']['basic']['login_strict']:
            sessdata = request.sessdata or config.get('SESSDATA')
            if not sessdata:
                raise HTTPException(
                    status_code=401,
                    detail="未登录：当前设置要求必须登录才能下载视频"
                )
            
            # 验证 SESSDATA 是否有效
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Cookie': f'SESSDATA={sessdata}'
            }
            
            response = requests.get(
                'https://api.bilibili.com/x/web-interface/nav',
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get('code') != 0:
                raise HTTPException(
                    status_code=401,
                    detail="登录已失效：请重新登录"
                )
        
        # 获取 yutto 可执行文件路径
        if getattr(sys, 'frozen', False):
            # 如果是打包后的 exe 运行
            base_path = os.path.dirname(sys.executable)
            paths_to_try = [
                os.path.join(base_path, 'yutto.exe'),  # 尝试主目录
                os.path.join(base_path, '_internal', 'yutto.exe'),  # 尝试 _internal 目录
                os.path.join(os.getcwd(), 'yutto.exe'),  # 尝试当前工作目录
                os.path.join(os.getcwd(), '_internal', 'yutto.exe')  # 尝试当前工作目录的 _internal
            ]
            
            yutto_path = None
            for path in paths_to_try:
                print(f"尝试路径：{path}")
                if os.path.exists(path):
                    yutto_path = path
                    print(f"找到 yutto.exe: {path}")
                    break
            
            if yutto_path is None:
                raise FileNotFoundError(f"找不到 yutto.exe，已尝试的路径：{', '.join(paths_to_try)}")
        else:
            # 如果是直接运行 python 脚本
            yutto_path = 'yutto'
            print(f"使用命令：{yutto_path}")

        if not os.path.exists(yutto_path) and getattr(sys, 'frozen', False):
            raise FileNotFoundError(f"找不到 yutto.exe，已尝试的路径：{yutto_path}")

        # 构建命令
        # 确保下载目录和临时目录存在且有正确的权限
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
        
        command = [
            yutto_path,
            request.url,
            '--dir', download_dir,
            '--tmp-dir', tmp_dir,
            '--subpath-template', f'{{title}}_{{username}}_{{download_date@%Y%m%d_%H%M%S}}_{request.cid}/{{title}}_{request.cid}',
            '--with-metadata'  # 添加元数据文件保存
        ]
        
        # 根据用户选择决定是否下载封面
        if not request.download_cover:
            command.append('--no-cover')
        
        # 根据用户选择决定是否仅下载音频
        if request.only_audio:
            command.append('--audio-only')
            command.extend(["--output-format-audio-only","flac"])
        
        # 添加其他 yutto 配置
        if not config['yutto']['resource']['require_subtitle']:
            command.append('--no-subtitle')
        
        if config['yutto']['danmaku']['font_size']:
            command.extend(['--danmaku-font-size', str(config['yutto']['danmaku']['font_size'])])
        
        if config['yutto']['batch']['with_section']:
            command.append('--with-section')
        
        # 如果提供了 SESSDATA，添加到命令中
        if request.sessdata:
            command.extend(['--sessdata', request.sessdata])
        elif config.get('SESSDATA'):
            command.extend(['--sessdata', config['SESSDATA']])

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

        # 在 Linux 上将命令列表转换为字符串，同时处理特殊字符
        command_str = None
        if sys.platform != 'win32':
            command_str = ' '.join(f"'{arg}'" if ((' ' in arg) or ("'" in arg) or ('"' in arg)) else arg for arg in command)

        try:
            # 执行命令，使用流式输出
            if sys.platform != 'win32':
                # 检查 FFmpeg 是否安装
                try:
                    ffmpeg_process = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
                    if ffmpeg_process.returncode != 0:
                        print("FFmpeg 未安装，需要手动安装...")
                        print(f"which ffmpeg 返回值：{ffmpeg_process.returncode}")
                        print(f"which ffmpeg 输出：{ffmpeg_process.stdout}")
                        print(f"which ffmpeg 错误：{ffmpeg_process.stderr}")
                        
                        # 检测系统类型
                        os_release = ""
                        try:
                            with open('/etc/os-release', 'r') as f:
                                os_release = f.read().lower()
                        except Exception as e:
                            print(f"读取/etc/os-release 失败：{str(e)}")
                        
                        # 准备安装指南
                        install_guide = "请按照以下步骤安装 FFmpeg:\n\n"
                        
                        if os.path.exists('/etc/centos-release') or 'centos' in os_release:
                            install_guide += "CentOS 7 安装步骤:\n\n"
                            install_guide += "1. 安装 EPEL 仓库:\n"
                            install_guide += "yum install -y epel-release\n\n"
                            install_guide += "2. 安装 RPM Fusion 仓库:\n"
                            install_guide += "yum localinstall -y --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm\n\n"
                            install_guide += "3. 安装 FFmpeg:\n"
                            install_guide += "yum install -y ffmpeg ffmpeg-devel"
                            
                        elif os.path.exists('/etc/debian_version') or 'ubuntu' in os_release or 'debian' in os_release:
                            install_guide += "Ubuntu/Debian 安装步骤:\n"
                            install_guide += "1. 更新包列表:\n"
                            install_guide += "apt-get update\n\n"
                            install_guide += "2. 安装 FFmpeg:\n"
                            install_guide += "apt-get install -y ffmpeg"
                            
                        else:
                            install_guide += "未能识别的 Linux 发行版，请访问 FFmpeg 官网获取安装指南：\n"
                            install_guide += "https://ffmpeg.org/download.html"
                        
                        raise HTTPException(
                            status_code=500,
                            detail=f"FFmpeg 未安装\n\n{install_guide}"
                        )
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"检查 FFmpeg 失败：{str(e)}\n请确保 FFmpeg 已正确安装并添加到系统 PATH 中"
                    )
                
                print(f"\n=== 执行命令 ===")
                print(f"命令：{command_str}")
                print(f"工作目录：{os.getcwd()}")
                print(f"yutto 路径：{yutto_path}")
                print(f"yutto 是否存在：{os.path.exists(yutto_path)}")
                print(f"yutto 是否可执行：{os.access(yutto_path, os.X_OK)}")
                print(f"\n环境变量：")
                for key, value in env.items():
                    print(f"{key}: {value}")
                
                # 检查 yutto 命令
                try:
                    version_process = subprocess.run(['yutto', '--version'], capture_output=True, text=True)
                    print(f"\nyutto 版本信息：")
                    print(version_process.stdout)
                    if version_process.stderr:
                        print(f"yutto 版本检查错误：{version_process.stderr}")
                except Exception as e:
                    print(f"检查 yutto 版本失败：{str(e)}")
                
                process = subprocess.Popen(
                    command_str,
                    **popen_kwargs
                )
            else:
                process = subprocess.Popen(
                    command,
                    **popen_kwargs
                )

            # 返回 SSE 响应
            return StreamingResponse(
                stream_process_output(process),
                media_type="text/event-stream"
            )
        except Exception as e:
            # 记录详细的错误信息
            error_msg = f"命令执行失败：{str(e)}\n"
            error_msg += f"命令：{command if sys.platform == 'win32' else (command_str if command_str else ' '.join(command))}\n"
            error_msg += f"环境变量:\n"
            error_msg += f"PATH: {env.get('PATH')}\n"
            error_msg += f"PYTHONPATH: {env.get('PYTHONPATH')}\n"
            error_msg += f"下载目录：{download_dir} (可写：{os.access(download_dir, os.W_OK)})\n"
            error_msg += f"临时目录：{tmp_dir} (可写：{os.access(tmp_dir, os.W_OK)})"
            
            # 添加系统信息
            import platform
            error_msg += f"\n\n系统信息:\n"
            error_msg += f"操作系统：{platform.system()} {platform.release()}\n"
            error_msg += f"Python 版本：{sys.version}\n"
            error_msg += f"工作目录：{os.getcwd()}\n"
            error_msg += f"yutto 路径：{yutto_path}\n"
            error_msg += f"yutto 是否存在：{os.path.exists(yutto_path)}\n"
            error_msg += f"yutto 是否可执行：{os.access(yutto_path, os.X_OK)}"
            
            # 添加 FFmpeg 信息
            try:
                ffmpeg_process = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
                error_msg += f"\n\nFFmpeg 信息:\n"
                if ffmpeg_process.returncode == 0:
                    ffmpeg_path = ffmpeg_process.stdout.strip()
                    error_msg += f"FFmpeg 路径：{ffmpeg_path}\n"
                    # 获取 FFmpeg 版本
                    version_process = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                    if version_process.returncode == 0:
                        error_msg += f"FFmpeg 版本：{version_process.stdout.splitlines()[0]}\n"
            except Exception as ffmpeg_error:
                error_msg += f"\n\nFFmpeg 检查失败：{str(ffmpeg_error)}"
            
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

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
        如果未安装，返回安装指南
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
        
        # FFmpeg 未安装，准备安装指南
        os_release = ""
        try:
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    os_release = f.read().lower()
        except Exception as e:
            print(f"读取/etc/os-release 失败：{str(e)}")
        
        install_guide = "请按照以下步骤安装 FFmpeg:\n\n"
        
        if system == 'darwin':  # macOS
            install_guide += "macOS 安装步骤:\n\n"
            install_guide += "1. 使用 Homebrew 安装:\n"
            install_guide += "brew install ffmpeg\n\n"
            install_guide += "如果没有安装 Homebrew，请先安装 Homebrew:\n"
            install_guide += '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            
        elif system == 'linux':
            if os.path.exists('/etc/centos-release') or 'centos' in os_release:
                install_guide += "CentOS 7 安装步骤:\n\n"
                install_guide += "1. 安装 EPEL 仓库:\n"
                install_guide += "yum install -y epel-release\n\n"
                install_guide += "2. 安装 RPM Fusion 仓库:\n"
                install_guide += "yum localinstall -y --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm\n\n"
                install_guide += "3. 安装 FFmpeg:\n"
                install_guide += "yum install -y ffmpeg ffmpeg-devel"
                
            elif os.path.exists('/etc/debian_version') or 'ubuntu' in os_release or 'debian' in os_release:
                install_guide += "Ubuntu/Debian 安装步骤:\n"
                install_guide += "1. 更新包列表:\n"
                install_guide += "apt-get update\n\n"
                install_guide += "2. 安装 FFmpeg:\n"
                install_guide += "apt-get install -y ffmpeg"
                
            else:
                install_guide += "未能识别的 Linux 发行版，请访问 FFmpeg 官网获取安装指南：\n"
                install_guide += "https://ffmpeg.org/download.html"
                
        elif system == 'windows':
            install_guide += "Windows 安装步骤:\n\n"
            install_guide += "1. 使用 Scoop 安装 (推荐):\n"
            install_guide += "scoop install ffmpeg\n\n"
            install_guide += "如果没有安装 Scoop，请先安装 Scoop:\n"
            install_guide += "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser\n"
            install_guide += 'irm get.scoop.sh | iex\n\n'
            install_guide += "2. 或者访问 FFmpeg 官网下载可执行文件:\n"
            install_guide += "https://ffmpeg.org/download.html#build-windows"
            
        else:
            install_guide += "未能识别的操作系统，请访问 FFmpeg 官网获取安装指南：\n"
            install_guide += "https://ffmpeg.org/download.html"
        
        return {
            "status": "error",
            "installed": False,
            "message": "FFmpeg 未安装",
            "install_guide": install_guide,
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
        # 检查登录状态
        if config['yutto']['basic']['login_strict']:
            sessdata = request.sessdata or config.get('SESSDATA')
            if not sessdata:
                raise HTTPException(
                    status_code=401,
                    detail="未登录：当前设置要求必须登录才能下载视频"
                )
            
            # 验证 SESSDATA 是否有效
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Cookie': f'SESSDATA={sessdata}'
            }
            
            response = requests.get(
                'https://api.bilibili.com/x/web-interface/nav',
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get('code') != 0:
                raise HTTPException(
                    status_code=401,
                    detail="登录已失效：请重新登录"
                )
        
        # 获取 yutto 可执行文件路径
        if getattr(sys, 'frozen', False):
            # 如果是打包后的 exe 运行
            base_path = os.path.dirname(sys.executable)
            paths_to_try = [
                os.path.join(base_path, 'yutto.exe'),  # 尝试主目录
                os.path.join(base_path, '_internal', 'yutto.exe'),  # 尝试 _internal 目录
                os.path.join(os.getcwd(), 'yutto.exe'),  # 尝试当前工作目录
                os.path.join(os.getcwd(), '_internal', 'yutto.exe')  # 尝试当前工作目录的 _internal
            ]
            
            yutto_path = None
            for path in paths_to_try:
                print(f"尝试路径：{path}")
                if os.path.exists(path):
                    yutto_path = path
                    print(f"找到 yutto.exe: {path}")
                    break
            
            if yutto_path is None:
                raise FileNotFoundError(f"找不到 yutto.exe，已尝试的路径：{', '.join(paths_to_try)}")
        else:
            # 如果是直接运行 python 脚本
            yutto_path = 'yutto'
            print(f"使用命令：{yutto_path}")

        if not os.path.exists(yutto_path) and getattr(sys, 'frozen', False):
            raise FileNotFoundError(f"找不到 yutto.exe，已尝试的路径：{yutto_path}")

        # 构建命令
        # 确保下载目录和临时目录存在且有正确的权限
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
        
        # 构建用户空间 URL
        user_space_url = f"https://space.bilibili.com/{request.user_id}/video"
        
        command = [
            yutto_path,
            user_space_url,
            '--batch',  # 批量下载
            '--dir', download_dir,
            '--tmp-dir', tmp_dir,
            '--subpath-template', f'{{username}}的全部投稿视频/{{title}}_{{download_date@%Y%m%d_%H%M%S}}/{{title}}',
            '--with-metadata'  # 添加元数据文件保存
        ]
        
        # 根据用户选择决定是否下载封面
        if not request.download_cover:
            command.append('--no-cover')
        
        # 根据用户选择决定是否仅下载音频
        if request.only_audio:
            command.append('--audio-only')
        
        # 添加其他 yutto 配置
        if not config['yutto']['resource']['require_subtitle']:
            command.append('--no-subtitle')
        
        if config['yutto']['danmaku']['font_size']:
            command.extend(['--danmaku-font-size', str(config['yutto']['danmaku']['font_size'])])
        
        if config['yutto']['batch']['with_section']:
            command.append('--with-section')
        
        # 如果提供了 SESSDATA，添加到命令中
        if request.sessdata:
            command.extend(['--sessdata', request.sessdata])
        elif config.get('SESSDATA'):
            command.extend(['--sessdata', config['SESSDATA']])

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

        # 在 Linux 上将命令列表转换为字符串，同时处理特殊字符
        command_str = None
        if sys.platform != 'win32':
            command_str = ' '.join(f"'{arg}'" if ((' ' in arg) or ("'" in arg) or ('"' in arg)) else arg for arg in command)
        
        # 执行命令
        print(f"执行下载命令：{' '.join(command) if sys.platform == 'win32' else command_str}")
        
        if sys.platform == 'win32':
            process = subprocess.Popen(command, **popen_kwargs)
        else:
            process = subprocess.Popen(command_str, **popen_kwargs)

        # 创建一个响应流
        return StreamingResponse(
            stream_process_output(process),
            media_type="text/event-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载过程出错：{str(e)}")

@router.post("/download_favorites", summary="下载用户收藏夹视频")
async def download_favorites(request: FavoriteDownloadRequest):
    """
    下载用户的收藏夹视频
    
    Args:
        request: 包含用户 ID、收藏夹 ID 和可选 SESSDATA 的请求对象
        注意：不提供收藏夹 ID 时，将下载所有收藏夹
    """
    try:
        # 检查登录状态
        sessdata = request.sessdata or config.get('SESSDATA')
        if not sessdata:
            raise HTTPException(
                status_code=401,
                detail="未登录：下载收藏夹必须提供 SESSDATA"
            )
        
        # 验证 SESSDATA 是否有效
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Cookie': f'SESSDATA={sessdata}'
        }
        
        response = requests.get(
            'https://api.bilibili.com/x/web-interface/nav',
            headers=headers,
            timeout=10
        )
        
        data = response.json()
        if data.get('code') != 0:
            raise HTTPException(
                status_code=401,
                detail="登录已失效：请重新登录"
            )
        
        # 获取 yutto 可执行文件路径
        if getattr(sys, 'frozen', False):
            # 如果是打包后的 exe 运行
            base_path = os.path.dirname(sys.executable)
            paths_to_try = [
                os.path.join(base_path, 'yutto.exe'),  # 尝试主目录
                os.path.join(base_path, '_internal', 'yutto.exe'),  # 尝试 _internal 目录
                os.path.join(os.getcwd(), 'yutto.exe'),  # 尝试当前工作目录
                os.path.join(os.getcwd(), '_internal', 'yutto.exe')  # 尝试当前工作目录的 _internal
            ]
            
            yutto_path = None
            for path in paths_to_try:
                print(f"尝试路径：{path}")
                if os.path.exists(path):
                    yutto_path = path
                    print(f"找到 yutto.exe: {path}")
                    break
            
            if yutto_path is None:
                raise FileNotFoundError(f"找不到 yutto.exe，已尝试的路径：{', '.join(paths_to_try)}")
        else:
            # 如果是直接运行 python 脚本
            yutto_path = 'yutto'
            print(f"使用命令：{yutto_path}")

        if not os.path.exists(yutto_path) and getattr(sys, 'frozen', False):
            raise FileNotFoundError(f"找不到 yutto.exe，已尝试的路径：{yutto_path}")

        # 构建命令
        # 确保下载目录和临时目录存在且有正确的权限
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
        
        # 构建收藏夹 URL
        if request.fid:
            # 指定收藏夹
            favorite_url = f"https://space.bilibili.com/{request.user_id}/favlist?fid={request.fid}"
        else:
            # 所有收藏夹
            favorite_url = f"https://space.bilibili.com/{request.user_id}/favlist"
        
        command = [
            yutto_path,
            favorite_url,
            '--batch',  # 批量下载
            '--dir', download_dir,
            '--tmp-dir', tmp_dir,
            '--subpath-template', f'{{username}}的收藏夹/{{title}}_{{download_date@%Y%m%d_%H%M%S}}/{{title}}',
            '--with-metadata'  # 添加元数据文件保存
        ]
        
        # 根据用户选择决定是否下载封面
        if not request.download_cover:
            command.append('--no-cover')
        
        # 根据用户选择决定是否仅下载音频
        if request.only_audio:
            command.append('--audio-only')
        
        # 添加其他 yutto 配置
        if not config['yutto']['resource']['require_subtitle']:
            command.append('--no-subtitle')
        
        if config['yutto']['danmaku']['font_size']:
            command.extend(['--danmaku-font-size', str(config['yutto']['danmaku']['font_size'])])
        
        if config['yutto']['batch']['with_section']:
            command.append('--with-section')
        
        # 添加 SESSDATA
        command.extend(['--sessdata', sessdata])

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

        # 在 Linux 上将命令列表转换为字符串，同时处理特殊字符
        command_str = None
        if sys.platform != 'win32':
            command_str = ' '.join(f"'{arg}'" if ((' ' in arg) or ("'" in arg) or ('"' in arg)) else arg for arg in command)
        
        # 执行命令
        print(f"执行下载命令：{' '.join(command) if sys.platform == 'win32' else command_str}")
        
        if sys.platform == 'win32':
            process = subprocess.Popen(command, **popen_kwargs)
        else:
            process = subprocess.Popen(command_str, **popen_kwargs)

        # 创建一个响应流
        return StreamingResponse(
            stream_process_output(process),
            media_type="text/event-stream"
        )

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
async def get_video_info(aid: Optional[int] = None, bvid: Optional[str] = None, sessdata: Optional[str] = None):
    """
    获取 B 站视频的详细信息
    
    Args:
        aid: 可选，视频的 avid
        bvid: 可选，视频的 bvid
        sessdata: 可选，用户的 SESSDATA，用于获取限制观看的视频
    
    Returns:
        视频详细信息
    """
    try:
        # 检查参数是否有效
        if not aid and not bvid:
            return VideoDetailResponse(
                status="error",
                message="必须提供 aid 或 bvid 参数中的至少一个"
            )
            
        # 设置请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        # 如果提供了 SESSDATA，添加到请求头中
        if sessdata:
            headers['Cookie'] = f'SESSDATA={sessdata}'
        elif config.get('SESSDATA'):
            headers['Cookie'] = f'SESSDATA={config["SESSDATA"]}'
            
        # 构建请求参数 - 确保 URL 编码正确
        url = 'https://api.bilibili.com/x/web-interface/view'
        
        if aid:
            url += f"?aid={aid}"
        elif bvid:
            # 对 bvid 进行 URL 编码，防止特殊字符引起问题
            import urllib.parse
            encoded_bvid = urllib.parse.quote(bvid)
            url += f"?bvid={encoded_bvid}"
            
        # 发送请求获取视频信息 - 不使用 params 参数，直接在 URL 中构建
        print(f"请求 URL: {url}")
        print(f"请求头：{headers}")
        
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )
        
        # 显式设置响应编码
        response.encoding = 'utf-8'
        
        # 打印响应状态和内容预览，便于调试
        print(f"响应状态码：{response.status_code}")
        content_preview = response.text[:100] if len(response.text) > 100 else response.text
        print(f"响应内容预览：{content_preview}")
        
        # 解析响应
        response_json = response.json()
        
        # 处理可能的错误
        if response_json.get('code') != 0:
            return VideoDetailResponse(
                status="error",
                message=f"获取视频信息失败：{response_json.get('message', '未知错误')}",
                data=response_json
            )
            
        # 返回成功响应
        return VideoDetailResponse(
            status="success",
            message="获取视频信息成功",
            data=response_json.get('data')
        )
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"获取视频信息时出错：{str(e)}")
        print(f"错误堆栈：{error_trace}")
        
        return VideoDetailResponse(
            status="error",
            message=f"获取视频信息时出错：{str(e)}",
            data={"error_trace": error_trace}
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
    sessdata: Optional[str] = None
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
        
        # 如果提供了 SESSDATA，添加到请求头中
        if sessdata:
            headers['Cookie'] = f'SESSDATA={sessdata}'
        elif config.get('SESSDATA'):
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