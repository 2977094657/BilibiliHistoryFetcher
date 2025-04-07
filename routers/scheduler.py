import asyncio
import logging
import os

import yaml
from fastapi import APIRouter, HTTPException, Depends, Request, Query

from scripts.scheduler_db_enhanced import EnhancedSchedulerDB
from scripts.scheduler_manager import SchedulerManager
from scripts.utils import get_config_path as utils_get_config_path

# 配置日志记录
logger = logging.getLogger(__name__)

router = APIRouter()

# 获取调度器实例
def get_scheduler():
    return SchedulerManager.get_instance()

# 获取调度器数据库实例
def get_scheduler_db():
    return EnhancedSchedulerDB.get_instance()

def get_config_path():
    """获取配置文件路径"""
    # 使用utils中的公共函数来保持一致性
    return utils_get_config_path('scheduler_config.yaml')

def save_config(config):
    """保存配置到文件"""
    config_path = get_config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return True

@router.get("/tasks", summary="获取任务信息")
async def get_tasks(
    task_id: str = None,
    include_subtasks: bool = True,
    detail_level: str = "basic",
    db: EnhancedSchedulerDB = Depends(get_scheduler_db)
):
    """
    获取任务信息
    
    - 不指定task_id时返回所有任务
    - 指定task_id时返回特定任务详情
    - include_subtasks控制是否包含子任务信息
    - detail_level控制返回信息详细程度
    """
    try:
        tasks = []
        if task_id:
            if db.is_main_task(task_id):
                task_data = db.get_main_task_by_id(task_id)
                if not task_data:
                    return {"status": "error", "message": f"任务 {task_id} 不存在"}
                if include_subtasks:
                    task_data['sub_tasks'] = db.get_sub_tasks(task_id)
                tasks.append(_build_task_info(task_data))
            else:
                task_data = db.get_subtask_by_id(task_id)
                if not task_data:
                    return {"status": "error", "message": f"任务 {task_id} 不存在"}
                tasks.append(_build_task_info(task_data))
        else:
            main_tasks = db.get_all_main_tasks()
            for task_data in main_tasks:
                if include_subtasks:
                    task_data['sub_tasks'] = db.get_sub_tasks(task_data['task_id'])
                tasks.append(_build_task_info(task_data))
        
        return {
            "status": "success",
            "message": "获取任务信息成功",
            "tasks": tasks
        }
        
    except Exception as e:
        logger.error(f"获取任务信息失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"获取任务信息失败: {str(e)}"
        }

def _build_task_info(task_data):
    """构建任务信息"""
    # 从配置文件中获取调度类型
    config_path = get_config_path()
    schedule_type = 'daily'  # 默认值
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                task_id = task_data.get('task_id')
                if task_id and 'tasks' in config and task_id in config['tasks']:
                    schedule_type = config['tasks'][task_id].get('schedule', {}).get('type', 'daily')
        except Exception as e:
            logger.error(f"读取配置文件失败: {str(e)}")
    
    config = {
        "name": task_data.get('name', ''),
        "endpoint": task_data.get('endpoint', ''),
        "method": task_data.get('method', 'GET'),
        "params": task_data.get('params', {}),
        "schedule_type": task_data.get('schedule_type') or schedule_type,
        "schedule_time": task_data.get('schedule_time'),
        "schedule_delay": task_data.get('schedule_delay'),
        "interval_value": task_data.get('interval_value'),
        "interval_unit": task_data.get('interval_unit'),
        "enabled": bool(task_data.get('enabled', True)),
        "priority": task_data.get('priority', 0),
        "tags": task_data.get('tags') or []
    }
    
    # 安全处理数值类型
    avg_duration = task_data.get('avg_duration')
    if avg_duration is not None:
        try:
            avg_duration = float(avg_duration)
        except (TypeError, ValueError):
            avg_duration = 0.0
    else:
        avg_duration = 0.0
    
    execution = {
        "last_run": task_data.get('last_run_time'),
        "next_run": task_data.get('next_run_time'),
        "status": task_data.get('last_status') or 'pending',
        "success_rate": task_data.get('success_rate', 0.0),
        "avg_duration": avg_duration,
        "total_runs": int(task_data.get('total_runs') or 0),
        "success_runs": int(task_data.get('success_runs') or 0),
        "fail_runs": int(task_data.get('fail_runs') or 0),
        "last_error": task_data.get('last_error')
    }
    
    task_type = task_data.get('task_type')
    if not task_type:
        task_type = 'sub' if task_data.get('parent_id') else 'main'
    
    result = {
        "task_id": task_data['task_id'],
        "task_type": task_type,
        "config": config,
        "execution": execution,
        "parent_id": task_data.get('parent_id'),
        "sub_tasks": [_build_subtask_info(sub_task) for sub_task in task_data.get('sub_tasks', [])] if task_data.get('sub_tasks') else None,
        "created_at": task_data.get('created_at'),
        "last_modified": task_data.get('last_modified')
    }
    
    # 添加依赖信息
    if task_data.get('depends_on'):
        result['depends_on'] = task_data['depends_on']
    
    return result

def _build_subtask_info(task_data):
    """构建子任务信息"""
    task_info = _build_task_info(task_data)
    return {
        "task_id": task_info["task_id"],
        "config": task_info["config"],
        "execution": task_info["execution"],
        "sequence_number": task_data.get('sequence_number', 0),
        "created_at": task_data.get('created_at'),
        "last_modified": task_data.get('last_modified')
    }

@router.post("/tasks", summary="创建新任务")
async def create_task(
    task_data: dict,
    db: EnhancedSchedulerDB = Depends(get_scheduler_db),
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    创建新任务
    
    - 支持创建主任务和子任务
    - 子任务需要指定parent_id和sequence_number
    """
    try:
        task_id = task_data.get('task_id')
        task_type = task_data.get('task_type', 'main')
        
        if task_type == "main":
            success = scheduler.add_main_task(task_id, task_data.get('config', {}))
        else:
            parent_id = task_data.get('parent_id')
            if not parent_id:
                return {"status": "error", "message": "创建子任务时必须指定parent_id"}
            success = scheduler.add_sub_task(parent_id, task_id, task_data.get('config', {}))
        
        if success:
            # 重新加载调度器配置
            scheduler.reload_scheduler()
            
            task_info = None
            if task_type == "main":
                task_info = _build_task_info(db.get_main_task_by_id(task_id))
            else:
                task_info = _build_task_info(db.get_subtask_by_id(task_id))
            
            return {
                "status": "success",
                "message": f"成功创建{task_type}任务",
                "task_id": task_id,
                "task_info": task_info
            }
        
        return {
            "status": "error",
            "message": "创建任务失败",
            "task_id": task_id
        }
            
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"创建任务失败: {str(e)}"
        }

@router.put("/tasks/{task_id}", summary="更新任务配置")
async def update_task(
    task_id: str,
    task_data: dict,
    db: EnhancedSchedulerDB = Depends(get_scheduler_db),
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    更新任务配置
    
    - 支持更新主任务和子任务
    - 支持更新任务配置、状态、优先级、标签等
    - 支持更新子任务执行顺序
    """
    try:
        if db.is_main_task(task_id):
            success = db.update_main_task(task_id, task_data)
        else:
            success = db.update_subtask(task_id, task_data)
        
        if success:
            # 重新加载调度器配置
            scheduler.reload_scheduler()
            
            task_info = None
            if db.is_main_task(task_id):
                task_info = _build_task_info(db.get_main_task_by_id(task_id))
            else:
                task_info = _build_task_info(db.get_subtask_by_id(task_id))
            
            return {
                "status": "success",
                "message": "任务更新成功",
                "task_id": task_id,
                "task_info": task_info
            }
        else:
            return {
                "status": "error",
                "message": "任务更新失败",
                "task_id": task_id
            }
            
    except Exception as e:
        logger.error(f"更新任务失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"更新任务失败: {str(e)}"
        }

@router.delete("/tasks/{task_id}", summary="删除任务")
async def delete_task(
    task_id: str,
    db: EnhancedSchedulerDB = Depends(get_scheduler_db)
):
    """
    删除任务
    
    - 支持删除主任务和子任务
    - 删除主任务时会自动删除其所有子任务
    """
    try:
        if db.is_main_task(task_id):
            success = db.delete_main_task(task_id)
        else:
            success = db.delete_subtask(task_id)
        
        if success:
            return {
                "status": "success",
                "message": "任务删除成功",
                "task_id": task_id
            }
        else:
            return {
                "status": "error",
                "message": "任务删除失败",
                "task_id": task_id
            }
            
    except Exception as e:
        logger.error(f"删除任务失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"删除任务失败: {str(e)}"
        }

@router.post("/tasks/{task_id}/execute", summary="执行任务")
async def execute_task(
    task_id: str,
    execute_params: dict = None,
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    执行任务
    
    - 支持执行主任务和子任务
    - 可控制是否执行子任务
    - 可选择是否等待任务完成
    """
    try:
        wait_for_completion = execute_params.get('wait_for_completion', False) if execute_params else False
        
        if wait_for_completion:
            success = await scheduler.execute_task(task_id)
        else:
            asyncio.create_task(scheduler.execute_task(task_id))
            success = True
        
        if success:
            return {
                "status": "success",
                "message": "任务执行已启动",
                "task_id": task_id
            }
        
        return {
            "status": "error",
            "message": "任务执行失败",
            "task_id": task_id
        }
            
    except Exception as e:
        logger.error(f"执行任务失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"执行任务失败: {str(e)}"
        }

@router.get("/tasks/history", summary="查询任务执行历史")
async def get_task_history(
    task_id: str = None,
    include_subtasks: bool = True,
    status: str = None,
    start_date: str = None,
    end_date: str = None,
    page: int = 1,
    page_size: int = 20,
    db: EnhancedSchedulerDB = Depends(get_scheduler_db)
):
    """
    查询任务执行历史
    
    - 支持查询主任务和子任务历史
    - 支持按时间范围和状态过滤
    - 支持分页查询
    """
    try:
        conditions = {
            'status': status,
            'start_date': start_date,
            'end_date': end_date
        }
        
        if task_id:
            # 无论是主任务还是子任务，都使用 get_task_execution_history_enhanced
            result = db.get_task_execution_history_enhanced(
                task_id,
                include_subtasks=include_subtasks if db.is_main_task(task_id) else False,
                conditions=conditions,
                page=page,
                page_size=page_size
            )
        else:
            result = db.get_recent_task_executions_enhanced(
                include_subtasks=include_subtasks,
                conditions=conditions,
                page=page,
                page_size=page_size
            )
        
        return {
            "status": "success",
            "message": "获取任务执行历史成功",
            "history": result['records'],
            "total_count": result['total'],
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"获取任务执行历史失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"获取任务执行历史失败: {str(e)}"
        }

@router.get("/available-endpoints", summary="获取可用API端点")
async def get_available_endpoints(request: Request):
    """
    获取所有可用的API端点信息
    
    - 返回系统中所有已注册的API端点
    - 包含路径、方法、描述等信息
    - 不包含系统内置的文档相关端点
    """
    try:
        # 需要过滤的内置端点路径
        builtin_paths = {
            '/openapi.json',
            '/docs',
            '/docs/oauth2-redirect',
            '/redoc',
            '/redoc-models.js'
        }
        
        endpoints = []
        for route in request.app.routes:
            if hasattr(route, 'methods'):
                # 跳过内置的文档相关端点
                if route.path in builtin_paths:
                    continue
                    
                # 跳过HEAD方法（通常是自动生成的）
                methods = [m for m in route.methods if m != 'HEAD']
                if not methods:
                    continue
                    
                endpoints.append({
                    "path": route.path,
                    "method": methods[0],
                    "summary": getattr(route, 'summary', None),
                    "tags": getattr(route, 'tags', []),
                    "operationId": route.name if hasattr(route, 'name') else None
                })
        
        # 按路径排序，使输出更有序
        endpoints.sort(key=lambda x: x['path'])
        
        return {
            "status": "success",
            "message": f"获取API端点列表成功，共 {len(endpoints)} 个端点",
            "total": len(endpoints),
            "endpoints": endpoints
        }
        
    except Exception as e:
        logger.error(f"获取API端点列表失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"获取API端点列表失败: {str(e)}"
        }

@router.post("/tasks/{task_id}/subtasks", summary="添加子任务")
async def add_sub_task(
    task_id: str,
    sub_task_data: dict,
    scheduler: SchedulerManager = Depends(get_scheduler),
    db: EnhancedSchedulerDB = Depends(get_scheduler_db)
):
    """
    为主任务添加子任务
    
    - 需要指定子任务名称、执行顺序、API端点等信息
    - 自动设置子任务为启用状态
    - 从配置文件读取调度类型，默认为'daily'
    """
    try:
        if task_id not in scheduler.tasks:
            return {"status": "error", "message": f"主任务 {task_id} 不存在"}
        
        if not db.is_main_task(task_id):
            return {"status": "error", "message": f"任务 {task_id} 不是主任务"}
        
        # 从配置文件获取调度类型
        config_path = get_config_path()
        schedule_type = 'daily'  # 默认值
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    sub_task_id = sub_task_data.get('task_id')
                    if sub_task_id and 'tasks' in config and sub_task_id in config['tasks']:
                        schedule_type = config['tasks'][sub_task_id].get('schedule', {}).get('type', 'daily')
            except Exception as e:
                logger.error(f"读取配置文件失败: {str(e)}")
        
        sub_task_config = {
            "task_id": sub_task_data.get('task_id'),
            "name": sub_task_data.get('name'),
            "sequence_number": sub_task_data.get('sequence_number'),
            "endpoint": sub_task_data.get('endpoint'),
            "method": sub_task_data.get('method', 'GET'),
            "params": sub_task_data.get('params'),
            "enabled": True,
            "task_type": "sub",
            "schedule_type": sub_task_data.get('schedule_type') or schedule_type,
            "depends_on": sub_task_data.get('depends_on')  # 添加依赖关系
        }
        
        # 获取子任务ID
        sub_task_id = sub_task_data.get('task_id')
        if not sub_task_id:
            return {"status": "error", "message": "子任务ID不能为空"}
            
        result = scheduler.add_sub_task(task_id, sub_task_id, sub_task_config)
        
        if not result:
            return {"status": "error", "message": "添加子任务失败"}
            
        # 重新加载调度器配置
        scheduler.reload_scheduler()
        
        return {
            "status": "success",
            "message": f"成功为主任务 {task_id} 添加子任务",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"添加子任务失败: {str(e)}")
        return {"status": "error", "message": f"添加子任务失败: {str(e)}"}

@router.get("/tasks/{task_id}/subtasks", summary="获取子任务列表")
async def get_sub_tasks(
    task_id: str, 
    db: EnhancedSchedulerDB = Depends(get_scheduler_db),
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    获取主任务的所有子任务
    
    - 返回子任务的详细信息，包括调度类型
    - 包含执行状态、成功率等统计数据
    - 按执行顺序排序
    """
    try:
        # 检查主任务是否存在
        if task_id not in scheduler.tasks:
            raise HTTPException(status_code=404, detail=f"主任务 {task_id} 不存在")
        
        # 检查是否为主任务
        if not db.is_main_task(task_id):
            raise HTTPException(status_code=400, detail=f"任务 {task_id} 不是主任务")
        
        # 获取子任务列表
        sub_tasks = db.get_sub_tasks(task_id)
        
        # 处理每个子任务的状态
        processed_sub_tasks = []
        for sub_task in sub_tasks:
            sub_task_id = sub_task['task_id']
            
            # 获取子任务状态
            status = '未执行'
            if sub_task.get('last_run_time'):
                status = '成功' if sub_task.get('last_status') == 'success' else '失败'
            
            # 计算成功率
            total_runs = sub_task.get('total_runs', 0)
            success_runs = sub_task.get('success_runs', 0)
            success_rate = round(success_runs / total_runs * 100, 2) if total_runs > 0 else None
            
            processed_sub_task = {
                'task_id': sub_task_id,
                'name': sub_task.get('name', sub_task_id),
                'sequence_number': sub_task.get('sequence_number', 0),
                'endpoint': sub_task.get('endpoint'),
                'method': sub_task.get('method', 'GET'),
                'params': sub_task.get('params'),
                'status': status,
                'enabled': bool(sub_task.get('enabled', True)),
                'success_rate': success_rate,
                'total_runs': total_runs,
                'success_runs': success_runs,
                'fail_runs': sub_task.get('fail_runs', 0),
                'last_error': sub_task.get('last_error'),
                'tags': sub_task.get('tags', []),
                'task_type': 'sub',
                'schedule_type': sub_task.get('schedule_type', 'daily')
            }
            processed_sub_tasks.append(processed_sub_task)
        
        # 按序号排序
        processed_sub_tasks.sort(key=lambda x: x['sequence_number'])
        
        return {
            "status": "success",
            "message": f"成功获取主任务 {task_id} 的子任务列表",
            "sub_tasks": processed_sub_tasks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取子任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取子任务列表失败: {str(e)}")

@router.delete("/tasks/{task_id}/subtasks/{sub_task_id}", summary="删除子任务")
async def delete_sub_task(
    task_id: str, 
    sub_task_id: str,
    db: EnhancedSchedulerDB = Depends(get_scheduler_db),
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    删除子任务
    
    - 从主任务中移除指定的子任务
    - 同时删除子任务的执行历史记录
    """
    try:
        # 检查主任务是否存在
        if task_id not in scheduler.tasks:
            raise HTTPException(status_code=404, detail=f"主任务 {task_id} 不存在")
        
        # 检查是否为主任务
        if not db.is_main_task(task_id):
            raise HTTPException(status_code=400, detail=f"任务 {task_id} 不是主任务")
        
        # 检查子任务是否存在
        sub_task = db.get_sub_task(task_id, sub_task_id)
        if not sub_task:
            raise HTTPException(status_code=404, detail=f"子任务 {sub_task_id} 不存在")
        
        # 删除子任务
        result = db.delete_sub_task(task_id, sub_task_id)
        
        if not result:
            raise HTTPException(status_code=500, detail="删除子任务失败")
        
        return {
            "status": "success",
            "message": f"成功删除子任务 {sub_task_id}",
            "task_id": task_id,
            "sub_task_id": sub_task_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除子任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除子任务失败: {str(e)}")

@router.put("/tasks/{task_id}/subtasks/{sub_task_id}/sequence", summary="更新子任务顺序")
async def update_sub_task_sequence(
    task_id: str,
    sub_task_id: str,
    sequence: int = Query(..., description="新的执行顺序"),
    db: EnhancedSchedulerDB = Depends(get_scheduler_db),
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    更新子任务的执行顺序
    
    - 修改子任务在主任务中的执行顺序
    - 自动调整其他子任务的顺序
    """
    try:
        # 检查主任务是否存在
        if task_id not in scheduler.tasks:
            raise HTTPException(status_code=404, detail=f"主任务 {task_id} 不存在")
        
        # 检查是否为主任务
        if not db.is_main_task(task_id):
            raise HTTPException(status_code=400, detail=f"任务 {task_id} 不是主任务")
        
        # 检查子任务是否存在
        sub_task = db.get_sub_task(task_id, sub_task_id)
        if not sub_task:
            raise HTTPException(status_code=404, detail=f"子任务 {sub_task_id} 不存在")
        
        # 更新执行顺序
        result = db.update_sub_task_sequence(task_id, sub_task_id, sequence)
        
        if not result:
            raise HTTPException(status_code=500, detail="更新子任务执行顺序失败")
        
        return {
            "status": "success",
            "message": f"成功更新子任务 {sub_task_id} 的执行顺序为 {sequence}",
            "task_id": task_id,
            "sub_task_id": sub_task_id,
            "sequence": sequence
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新子任务执行顺序失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新子任务执行顺序失败: {str(e)}")

@router.post("/tasks/{task_id}/enable", summary="启用/禁用任务")
async def enable_task(
    task_id: str,
    request: dict,
    db: EnhancedSchedulerDB = Depends(get_scheduler_db),
    scheduler: SchedulerManager = Depends(get_scheduler)
):
    """
    启用或禁用任务
    
    - 支持启用/禁用主任务和子任务
    - 禁用主任务时会自动禁用其所有子任务
    - 启用主任务不会自动启用子任务
    - 子任务的启用/禁用不影响其他任务
    """
    try:
        # 检查任务是否存在
        is_main = db.is_main_task(task_id)
        task = None
        
        if is_main:
            task = db.get_main_task_by_id(task_id)
        else:
            task = db.get_subtask_by_id(task_id)
            
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        enabled = request.get('enabled', True)
        
        if is_main:
            # 更新主任务状态
            result = db.update_main_task(task_id, {'enabled': enabled})
            
            if enabled is False:
                # 如果是禁用主任务，同时禁用所有子任务
                sub_tasks = db.get_sub_tasks(task_id)
                for sub_task in sub_tasks:
                    db.update_subtask(sub_task['task_id'], {'enabled': False})
                
                success_message = f"已禁用主任务 {task_id} 及其所有子任务"
            else:
                success_message = f"已启用主任务 {task_id}"
        else:
            # 更新子任务状态
            result = db.update_subtask(task_id, {'enabled': enabled})
            success_message = f"已{'启用' if enabled else '禁用'}子任务 {task_id}"
        
        if not result:
            raise HTTPException(status_code=500, detail=f"更新任务状态失败")
        
        return {
            "status": "success",
            "message": success_message,
            "task_id": task_id,
            "enabled": enabled
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新任务状态失败: {str(e)}") 