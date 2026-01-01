from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncio
import time
from datetime import datetime

from scripts.popular_videos import (
    get_all_popular_videos, 
    query_recent_videos,
    get_fetch_history,
    get_video_tracking_stats,
    get_all_year_dbs,
    cleanup_inactive_video_records
)

# 创建路由器
router = APIRouter()

class PopularVideosResponse(BaseModel):
    status: str
    message: Optional[str] = None
    total_videos: Optional[int] = None
    fetch_stats: Optional[Dict[str, Any]] = None
    data: Optional[List[Dict[str, Any]]] = None
    task_id: Optional[str] = None

class FetchHistoryResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None

class TrackingStatsResponse(BaseModel):
    status: str
    message: Optional[str] = None
    total: Optional[int] = None
    data: Optional[List[Dict[str, Any]]] = None

class PopularYearsResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: List[int] = Field(default_factory=list, description="热门视频数据库可用年份（降序）")
    default_year: Optional[int] = Field(default=None, description="建议默认选择年份（最新年份）")

# 用于存储后台任务的状态和清理机制
task_status = {}
MAX_TASKS_HISTORY = 100  # 最多保存的任务数量
TASK_EXPIRY_TIME = 3600  # 任务过期时间（秒）

# 清理过期任务
def cleanup_old_tasks():
    """清理过期的任务"""
    current_time = time.time()
    expired_tasks = []
    
    for task_id, task_info in task_status.items():
        # 检查任务是否有时间戳，如果没有则添加
        if "timestamp" not in task_info:
            task_info["timestamp"] = current_time
            continue
            
        # 检查任务是否已完成且超过过期时间
        if (task_info["status"] in ["completed", "failed"] and 
            current_time - task_info["timestamp"] > TASK_EXPIRY_TIME):
            expired_tasks.append(task_id)
    
    # 删除过期任务
    for task_id in expired_tasks:
        del task_status[task_id]
    
    # 如果任务数量过多，按时间戳删除最旧的任务
    if len(task_status) > MAX_TASKS_HISTORY:
        # 按时间戳排序
        sorted_tasks = sorted(
            task_status.items(), 
            key=lambda x: x[1].get("timestamp", 0)
        )
        
        # 计算需要删除的任务数量
        tasks_to_remove = len(task_status) - MAX_TASKS_HISTORY
        
        # 删除最旧的任务
        for i in range(tasks_to_remove):
            if i < len(sorted_tasks):
                del task_status[sorted_tasks[i][0]]

@router.get("/popular/all", summary="获取B站所有热门视频", response_model=PopularVideosResponse)
async def get_all_popular_videos_api(
    background_tasks: BackgroundTasks,
    size: int = Query(20, description="每页视频数量", ge=1, le=50),
    max_pages: int = Query(100, description="最大获取页数", ge=1, le=500),
    save_to_db: bool = Query(True, description="是否保存到数据库"),
    include_videos: bool = Query(False, description="是否在响应中包含视频数据")
):
    """
    获取B站当前所有热门视频并保存到数据库（异步后台处理）
    
    此接口会立即返回一个任务ID，然后在后台处理获取视频的请求。
    使用返回的任务ID通过 `/popular/task/{task_id}` 端点查询任务状态和结果。
    
    - **size**: 每页视频数量，默认为20，范围1-50
    - **max_pages**: 最大获取页数，默认为100，范围1-500
    - **save_to_db**: 是否保存到数据库，默认为True
    - **include_videos**: 是否在响应中包含视频数据，默认为False
    
    返回:
    - **status**: 请求状态，"accepted"表示已接受任务
    - **message**: 状态消息
    - **task_id**: 用于查询任务状态的唯一ID
    """
    # 清理过期任务
    cleanup_old_tasks()
    
    # 生成唯一任务ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # 创建初始状态响应
    task_status[task_id] = {
        "status": "processing",
        "message": "任务已开始处理",
        "progress": 0,
        "result": None,
        "timestamp": time.time()
    }
    
    # 定义后台任务函数
    async def process_popular_videos_task():
        try:
            # 定义进度回调函数
            def update_progress(progress, message, current_page, total_pages, success=True):
                nonlocal task_id
                task_status[task_id].update({
                    "status": "processing" if success else "error",
                    "message": message,
                    "progress": progress,
                    "current_page": current_page,
                    "total_pages": total_pages,
                    "timestamp": time.time()  # 更新时间戳
                })
            
            # 在后台线程中执行耗时操作
            loop = asyncio.get_event_loop()
            videos, success, fetch_stats = await loop.run_in_executor(
                None, 
                lambda: get_all_popular_videos(
                    page_size=size, 
                    max_pages=max_pages, 
                    save_to_db=save_to_db,
                    progress_callback=update_progress
                )
            )
            
            # 构建结果
            result = {
                "status": "success" if success else "error",
                "total_videos": len(videos),
                "fetch_stats": fetch_stats["fetch_stats"] if "fetch_stats" in fetch_stats else fetch_stats
            }
            
            # 如果需要包含视频数据
            if include_videos and videos:
                result["data"] = videos
            
            if not success:
                result["message"] = fetch_stats.get("message", "获取热门视频失败")
            
            # 更新任务状态为完成
            task_status[task_id] = {
                "status": "completed" if success else "failed",
                "progress": 100,
                "result": result,
                "timestamp": time.time(),
                "message": "热门视频获取完成" if success else fetch_stats.get("message", "获取热门视频失败")
            }
        except Exception as e:
            # 更新任务状态为失败
            task_status[task_id] = {
                "status": "failed",
                "message": f"处理任务失败: {str(e)}",
                "progress": 0,
                "result": None,
                "timestamp": time.time()
            }
    
    # 添加到后台任务
    background_tasks.add_task(process_popular_videos_task)
    
    # 立即返回响应，不等待任务完成
    return PopularVideosResponse(
        status="accepted",
        message="任务已开始处理，请使用任务ID查询进度",
        task_id=task_id
    )

@router.get("/popular/task/{task_id}", summary="获取热门视频获取任务状态")
async def get_task_status(task_id: str):
    """
    获取热门视频获取任务的处理状态和结果
    
    使用从 `/popular/all` 接口获取的任务ID查询任务进度和结果。
    
    - 如果任务仍在处理中，将返回进度信息
    - 如果任务已完成，将返回完整的结果
    - 如果任务失败，将返回错误信息
    
    任务状态会在服务器上保留一段时间（默认1小时），过期后会被自动清理。
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="未找到指定任务ID")
    
    task_info = task_status[task_id]
    
    # 如果任务完成且有结果，返回结果
    if task_info["status"] == "completed" and task_info["result"]:
        return task_info["result"]
    
    # 否则返回任务状态
    return {
        "status": task_info["status"],
        "message": task_info.get("message", "任务正在处理中"),
        "progress": task_info.get("progress", 0)
    }

@router.get("/popular/recent", summary="获取最近保存的热门视频", response_model=PopularVideosResponse)
async def get_recent_popular_videos_api(
    limit: int = Query(20, description="返回视频的数量限制", ge=1, le=200)
):
    """
    从数据库获取最近保存的热门视频
    
    - **limit**: 返回视频的数量限制，默认为20
    """
    try:
        # 从数据库查询最近的热门视频
        videos = query_recent_videos(limit=limit)
        
        if not videos:
            return PopularVideosResponse(
                status="success",
                message="未找到任何热门视频记录",
                total_videos=0,
                data=[]
            )
        
        return PopularVideosResponse(
            status="success",
            total_videos=len(videos),
            data=videos
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取热门视频记录失败: {str(e)}"
        )

@router.get("/popular/history", summary="获取热门视频抓取历史", response_model=FetchHistoryResponse)
async def get_fetch_history_api(
    limit: int = Query(10, description="返回历史记录的数量限制", ge=1, le=100)
):
    """
    获取热门视频的抓取历史记录
    
    - **limit**: 返回历史记录的数量限制，默认为10
    """
    try:
        # 获取抓取历史记录
        history = get_fetch_history(limit=limit)
        
        if not history:
            return FetchHistoryResponse(
                status="success",
                message="未找到任何抓取历史记录",
                data=[]
            )
        
        return FetchHistoryResponse(
            status="success",
            data=history
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取抓取历史记录失败: {str(e)}"
        )

@router.get("/popular/tracking", summary="获取视频热门持续时间统计", response_model=TrackingStatsResponse)
async def get_video_tracking_stats_api(
    limit: int = Query(20, description="返回视频的数量限制", ge=1, le=100)
):
    """
    获取视频在热门列表中持续时间的统计信息
    
    - **limit**: 返回视频的数量限制，默认为20
    """
    try:
        # 获取视频热门跟踪统计
        stats = get_video_tracking_stats(limit=limit)
        
        if not stats:
            return TrackingStatsResponse(
                status="success",
                message="未找到任何视频热门跟踪记录",
                total=0,
                data=[]
            )
        
        return TrackingStatsResponse(
            status="success",
            total=len(stats),
            data=stats
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取视频热门跟踪统计失败: {str(e)}"
        )

@router.get("/popular/years", summary="获取热门视频数据库可用年份", response_model=PopularYearsResponse)
async def get_popular_db_years_api():
    """
    获取热门视频按年分库的可用年份列表（用于前端下拉菜单）
    """
    try:
        years = sorted(get_all_year_dbs(), reverse=True)
        if not years:
            return PopularYearsResponse(
                status="success",
                message="未找到任何热门视频数据库文件",
                data=[],
                default_year=None
            )

        return PopularYearsResponse(
            status="success",
            data=years,
            default_year=years[0]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取热门视频数据库年份失败: {str(e)}"
        )

@router.get("/popular/tasks", summary="获取所有热门视频获取任务")
async def list_all_tasks():
    """
    获取所有热门视频获取任务的状态列表
    
    返回当前服务器上所有正在处理中和已完成的热门视频获取任务，
    包括每个任务的ID、状态、进度和创建时间等信息。
    
    可以使用返回的任务ID通过 `/popular/task/{task_id}` 端点查询具体任务的详细信息。
    """
    # 清理过期任务
    cleanup_old_tasks()
    
    # 返回任务基本信息
    tasks_info = {}
    for task_id, task_info in task_status.items():
        task_summary = {
            "status": task_info["status"],
            "message": task_info.get("message", ""),
            "progress": task_info.get("progress", 0),
            "timestamp": task_info.get("timestamp", 0),
            "created": datetime.fromtimestamp(
                task_info.get("timestamp", 0)
            ).strftime("%Y-%m-%d %H:%M:%S") if task_info.get("timestamp") else ""
        }
        tasks_info[task_id] = task_summary
    
    return {
        "status": "success",
        "total_tasks": len(tasks_info),
        "tasks": tasks_info
    }

@router.post("/popular/cleanup", summary="清理已经不在热门列表的视频数据")
async def trigger_data_cleanup(
    background_tasks: BackgroundTasks,
    year: Optional[int] = Query(None, description="要清理的年份，不传则清理所有年份")
):
    """
    触发清理已经不在热门列表的视频数据的操作
    
    此操作会删除所有已经不在热门列表的视频中间的记录，只保留首条和末条记录。
    由于清理操作可能需要较长时间，此API会在后台执行清理并立即返回。
    清理结果将记录在日志中，也可通过任务ID查询结果。

    - **year**: 指定清理的年份；不传则清理所有年份数据库
    """
    available_years = sorted(get_all_year_dbs(), reverse=True)
    if year is not None and year not in available_years:
        return {
            "status": "error",
            "message": f"未找到 {year} 年的热门视频数据库。可用年份：{', '.join(map(str, available_years)) if available_years else '无'}",
            "available_years": available_years
        }

    # 清理过期任务
    cleanup_old_tasks()

    # 生成唯一任务ID
    import uuid
    task_id = str(uuid.uuid4())

    # 创建初始状态响应
    task_status[task_id] = {
        "status": "processing",
        "message": "数据清理任务已开始处理",
        "progress": 0,
        "result": None,
        "timestamp": time.time(),
        "meta": {
            "task_type": "cleanup",
            "year": year,
            "available_years": available_years,
            "target_years": [year] if year is not None else available_years
        }
    }

    # 定义后台任务函数
    async def run_cleanup_task():
        # 在异步上下文中执行同步耗时操作
        loop = asyncio.get_event_loop()
        try:
            # 使用run_in_executor在线程池中执行同步操作
            stats = await loop.run_in_executor(None, cleanup_inactive_video_records, year)
            print(f"数据清理完成，统计信息：{stats}")

            task_status[task_id] = {
                "status": "completed",
                "message": "数据清理完成",
                "progress": 100,
                "timestamp": time.time(),
                "result": {
                    "status": "success",
                    "message": "数据清理完成",
                    "year": year,
                    "available_years": available_years,
                    "data": stats
                },
                "meta": task_status[task_id].get("meta", {})
            }
        except Exception as e:
            error_message = f"数据清理过程中出错：{str(e)}"
            print(error_message)
            task_status[task_id] = {
                "status": "failed",
                "message": error_message,
                "progress": 0,
                "timestamp": time.time(),
                "result": {
                    "status": "error",
                    "message": error_message,
                    "year": year,
                    "available_years": available_years
                },
                "meta": task_status[task_id].get("meta", {})
            }
    
    # 添加到后台任务
    background_tasks.add_task(run_cleanup_task)
    
    # 立即返回响应
    return {
        "status": "accepted",
        "message": "数据清理任务已开始，将在后台执行。可通过 /bilibili/popular/task/{task_id} 查询状态。",
        "task_id": task_id,
        "year": year,
        "available_years": available_years
    } 
