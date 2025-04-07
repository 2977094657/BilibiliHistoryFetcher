import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from scripts.image_downloader import ImageDownloader
from scripts.utils import get_output_path

router = APIRouter()
downloader = ImageDownloader()

def get_history_db():
    """获取历史记录数据库连接"""
    db_path = get_output_path('bilibili_history.db')
    return sqlite3.connect(db_path)

@router.post("/start", summary="开始下载图片")
async def start_download(
    background_tasks: BackgroundTasks,
    year: Optional[int] = None
):
    """开始下载图片
    
    Args:
        year: 指定年份，不指定则下载所有年份
    """
    # 包装下载函数和状态更新
    def download_with_status_update(year=None):
        try:
            # 执行下载
            downloader.start_download(year)
        except Exception as e:
            print(f"下载过程发生错误: {str(e)}")
        finally:
            # 确保无论下载成功还是失败，状态都会被设置为已完成
            print("\n=== 下载任务完成，更新状态 ===")
            downloader.is_downloading = False
            print("下载状态已设置为已完成")
    
    # 在后台任务中执行包装函数
    background_tasks.add_task(download_with_status_update, year)
    
    return {
        "status": "success",
        "message": f"开始下载{'所有年份' if year is None else f'{year}年'}的图片"
    }

@router.post("/stop", summary="停止下载图片")
async def stop_download():
    """停止当前下载任务
    
    Returns:
        dict: 包含停止状态和当前下载统计的响应
    """
    try:
        result = downloader.stop_download()
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"停止下载失败: {str(e)}"
        }

@router.get("/status", summary="获取下载状态")
async def get_status():
    """获取下载状态"""
    stats = downloader.get_download_stats()
    
    return {
        "status": "success",
        "data": stats
    }

@router.post("/clear", summary="清空所有图片")
async def clear_images():
    """清空所有图片和下载状态"""
    try:
        success = downloader.clear_all_images()
        if success:
            return {
                "status": "success",
                "message": "已清空所有图片和下载状态",
                "data": {
                    "cleared_paths": [
                        "output/images/covers",
                        "output/images/avatars",
                        "output/images/orphaned_covers",
                        "output/images/orphaned_avatars"
                    ],
                    "status_file": "output/download_status.json"
                }
            }
        else:
            return {
                "status": "error",
                "message": "清空图片失败，请查看日志了解详细信息"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"清空图片时发生错误: {str(e)}"
        }

@router.get("/local/{image_type}/{file_hash}", summary="获取本地图片")
async def get_local_image(image_type: str, file_hash: str):
    """获取本地图片
    
    Args:
        image_type: 图片类型 (covers 或 avatars)
        file_hash: 图片文件的哈希值
        
    Returns:
        FileResponse: 图片文件响应
    """
    print(f"\n=== 获取本地图片 ===")
    print(f"图片类型: {image_type}")
    print(f"文件哈希: {file_hash}")
    
    # 验证图片类型
    if image_type not in ('covers', 'avatars'):
        raise HTTPException(
            status_code=400,
            detail=f"无效的图片类型: {image_type}"
        )
    
    try:
        # 构建图片路径
        base_path = get_output_path('images')
        type_path = os.path.join(base_path, image_type)
        sub_dir = file_hash[:2]  # 使用哈希的前两位作为子目录
        
        # 获取所有年份目录
        years = []
        if os.path.exists(type_path):
            for item in os.listdir(type_path):
                if item.isdigit():
                    years.append(item)
        
        # 按年份倒序搜索图片
        for year in sorted(years, reverse=True):
            year_path = os.path.join(type_path, year)
            img_dir = os.path.join(year_path, sub_dir)
            
            if not os.path.exists(img_dir):
                continue
                
            # 查找所有可能的图片文件扩展名
            for ext in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
                img_path = os.path.join(img_dir, f"{file_hash}{ext}")
                if os.path.exists(img_path):
                    print(f"找到图片文件: {img_path}")
                    return FileResponse(
                        img_path,
                        media_type=f"image/{ext[1:]}" if ext != '.jpg' else "image/jpeg"
                    )
        
        # 如果在年份目录中没有找到，尝试在根目录中查找
        img_dir = os.path.join(type_path, sub_dir)
        if os.path.exists(img_dir):
            for ext in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
                img_path = os.path.join(img_dir, f"{file_hash}{ext}")
                if os.path.exists(img_path):
                    print(f"找到图片文件: {img_path}")
                    return FileResponse(
                        img_path,
                        media_type=f"image/{ext[1:]}" if ext != '.jpg' else "image/jpeg"
                    )
        
        # 如果没有找到任何匹配的文件
        raise HTTPException(
            status_code=404,
            detail=f"图片不存在: {file_hash}"
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"获取本地图片时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取图片失败: {str(e)}"
        ) 