import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import calendar

import yaml
from dateutil.relativedelta import relativedelta

from scripts.scheduler_db import SchedulerDB

logger = logging.getLogger(__name__)

class EnhancedSchedulerDB(SchedulerDB):
    """增强版调度器数据库管理类，支持主次任务关系"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'EnhancedSchedulerDB':
        """获取单例实例"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """初始化数据库连接"""
        if hasattr(self, '_initialized'):
            return
            
        # 确保输出目录存在
        self.db_dir = os.path.join('output', 'database')
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.db_path = os.path.join(self.db_dir, 'scheduler.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 连接数据库并设置时区为 UTC+8
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA timezone='+08:00'")
        cursor = self.conn.cursor()
        cursor.execute("SELECT datetime('now', 'localtime')")
        self.conn.row_factory = sqlite3.Row
        
        # 如果数据库尚未初始化，则执行初始化操作
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            # 创建新的表结构
            self._create_tables()
            
            # 从配置文件导入初始数据
            self._import_config_data()
        
        self._initialized = True
    
    def _drop_all_tables(self):
        """删除所有现有表"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            if table[0] not in ['sqlite_sequence']:  # 保留系统表
                cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
        
        self.conn.commit()
    
    def _create_tables(self):
        """创建新的表结构"""
        cursor = self.conn.cursor()
        
        # 主任务表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS main_tasks (
            task_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            method TEXT DEFAULT 'GET',
            params TEXT,
            schedule_type TEXT NOT NULL,
            schedule_time TEXT,
            schedule_delay INTEGER,
            interval_value INTEGER,
            interval_unit TEXT,
            enabled INTEGER DEFAULT 1,
            task_type TEXT DEFAULT 'main',
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            last_modified TIMESTAMP DEFAULT (datetime('now', 'localtime'))
        )
        ''')
        
        # 子任务表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sub_tasks (
            task_id TEXT PRIMARY KEY,
            parent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            sequence_number INTEGER NOT NULL,
            endpoint TEXT NOT NULL,
            method TEXT DEFAULT 'GET',
            params TEXT,
            schedule_type TEXT DEFAULT 'daily',
            enabled INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            last_modified TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (parent_id) REFERENCES main_tasks(task_id) ON DELETE CASCADE
        )
        ''')
            
        # 任务状态表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_status (
            task_id TEXT PRIMARY KEY,
            last_run_time TEXT,
            next_run_time TEXT,
            last_status TEXT,
            total_runs INTEGER DEFAULT 0,
            success_runs INTEGER DEFAULT 0,
            fail_runs INTEGER DEFAULT 0,
            avg_duration REAL DEFAULT 0,
            last_error TEXT,
            tags TEXT,
            success_rate REAL DEFAULT 0,
            FOREIGN KEY (task_id) REFERENCES main_tasks(task_id) ON DELETE CASCADE
        )
        ''')
        
        # 子任务状态表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sub_task_status (
            task_id TEXT PRIMARY KEY,
            last_run_time TEXT,
            next_run_time TEXT,
            last_status TEXT,
            total_runs INTEGER DEFAULT 0,
            success_runs INTEGER DEFAULT 0,
            fail_runs INTEGER DEFAULT 0,
            avg_duration REAL DEFAULT 0,
            last_error TEXT,
            tags TEXT,
            success_rate REAL DEFAULT 0,
            FOREIGN KEY (task_id) REFERENCES sub_tasks(task_id) ON DELETE CASCADE
        )
        ''')
            
        # 任务执行历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration REAL,
            status TEXT NOT NULL,
            error_message TEXT,
            output TEXT,
            triggered_by TEXT,
            next_run_time TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (task_id) REFERENCES main_tasks(task_id) ON DELETE CASCADE
        )
        ''')
        
        # 子任务执行历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sub_task_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration REAL,
            status TEXT NOT NULL,
            error_message TEXT,
            output TEXT,
            triggered_by TEXT,
            next_run_time TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (task_id) REFERENCES sub_tasks(task_id) ON DELETE CASCADE
        )
        ''')
        
        # 任务依赖关系表 - 移除外键约束
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_dependencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            depends_on TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            UNIQUE(task_id, depends_on)
        )
        ''')
            
        self.conn.commit()
    
    def _import_config_data(self):
        """从配置文件导入初始数据"""
        try:
            from scripts.utils import get_config_path
            config_path = get_config_path('scheduler_config.yaml')
            
            if not os.path.exists(config_path):
                print(f"配置文件不存在: {config_path}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config or 'tasks' not in config:
                print("配置文件中没有任务数据")
                return
            
            cursor = self.conn.cursor()
            
            # 首先找出所有主任务（没有依赖的任务）
            main_tasks = {}
            sub_tasks = {}
            
            for task_id, task_data in config['tasks'].items():
                if not task_data.get('requires'):  # 没有依赖的是主任务
                    main_tasks[task_id] = task_data
                else:
                    sub_tasks[task_id] = task_data
            
            print(f"找到 {len(main_tasks)} 个主任务和 {len(sub_tasks)} 个子任务")
            
            # 导入主任务
            for task_id, task_data in main_tasks.items():
                schedule_info = task_data.get('schedule', {})
                main_task = {
                    'task_id': task_id,
                    'name': task_data.get('name', task_id),
                    'endpoint': task_data.get('endpoint', ''),
                    'method': task_data.get('method', 'GET'),
                    'params': json.dumps(task_data.get('params', {})),
                    'schedule_type': schedule_info.get('type', 'daily'),
                    'schedule_time': schedule_info.get('time'),
                    'schedule_delay': schedule_info.get('delay'),
                    'enabled': 1
                }
                
                # 插入主任务
                cursor.execute("""
                INSERT INTO main_tasks (
                    task_id, name, endpoint, method, params, schedule_type, 
                    schedule_time, schedule_delay, enabled, task_type, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id,
                    task_data.get('name', task_id),
                    task_data.get('endpoint', ''),
                    task_data.get('method', 'GET'),
                    json.dumps(task_data.get('params', {})),
                    schedule_info.get('type', 'daily'),
                    schedule_info.get('time'),
                    schedule_info.get('delay'),
                    task_data.get('enabled', 1),
                    'main',
                    datetime.now().isoformat()
                ))
                
                # 初始化任务状态
                cursor.execute('''
                INSERT INTO task_status (task_id, tags)
                VALUES (?, ?)
                ''', (task_id, json.dumps(task_data.get('tags', []))))
            
            # 导入子任务
            sequence_counter = {}  # 用于记录每个主任务的子任务序号
            
            for task_id, task_data in sub_tasks.items():
                # 找到父任务
                parent_id = self._find_root_task(task_id, config['tasks'])
                if not parent_id:
                    print(f"警告: 无法找到任务 {task_id} 的父任务")
                    continue
                
                # 初始化序号计数器
                if parent_id not in sequence_counter:
                    sequence_counter[parent_id] = 1
                
                schedule_info = task_data.get('schedule', {})
                sub_task = {
                    'task_id': task_id,
                    'parent_id': parent_id,
                    'name': task_data.get('name', task_id),
                    'sequence_number': sequence_counter[parent_id],
                    'endpoint': task_data.get('endpoint', ''),
                    'method': task_data.get('method', 'GET'),
                    'params': json.dumps(task_data.get('params', {})),
                    'schedule_type': schedule_info.get('type', 'daily'),  # 修改这里，默认为daily
                    'enabled': 1
                }
                
                # 插入子任务
                cursor.execute('''
                INSERT INTO sub_tasks (
                    task_id, parent_id, name, sequence_number,
                    endpoint, method, params, schedule_type, enabled
                ) VALUES (
                    :task_id, :parent_id, :name, :sequence_number,
                    :endpoint, :method, :params, :schedule_type, :enabled
                )
                ''', sub_task)
                
                # 初始化子任务状态
                cursor.execute('''
                INSERT INTO sub_task_status (task_id, tags)
                VALUES (?, ?)
                ''', (task_id, json.dumps(task_data.get('tags', []))))
                
                # 更新序号计数器
                sequence_counter[parent_id] += 1
                
                # 添加依赖关系
                for depends_on in task_data.get('requires', []):
                    cursor.execute('''
                    INSERT INTO task_dependencies (task_id, depends_on)
                    VALUES (?, ?)
                    ''', (task_id, depends_on))
            
            self.conn.commit()
            print("成功导入配置数据")
            
        except Exception as e:
            print(f"导入配置数据时出错: {str(e)}")
            self.conn.rollback()
    
    def _find_root_task(self, task_id: str, tasks: dict) -> str:
        """递归查找任务链的根任务（主任务）"""
        task_data = tasks.get(task_id)
        if not task_data or not task_data.get('requires'):
            return task_id
        return self._find_root_task(task_data['requires'][0], tasks)
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    # =================== 主任务管理 ===================
    
    def get_all_main_tasks(self) -> List[Dict]:
        """获取所有主任务"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT m.*, 
               ts.last_run_time, ts.next_run_time, ts.last_status,
               ts.total_runs, ts.success_runs, ts.fail_runs,
               ts.avg_duration, ts.last_error, ts.tags,
               ts.success_rate,
               m.created_at as created_at_local,
               m.last_modified as last_modified_local
        FROM main_tasks m
        LEFT JOIN task_status ts ON m.task_id = ts.task_id
        ''')
        
        rows = cursor.fetchall()
        result = []
        
        for row in rows:
            task_data = dict(row)
            # 处理JSON字段
            if task_data.get('params'):
                try:
                    task_data['params'] = json.loads(task_data['params'])
                except:
                    task_data['params'] = {}
            
            if task_data.get('tags'):
                try:
                    task_data['tags'] = json.loads(task_data['tags'])
                except:
                    task_data['tags'] = []
            
            # 使用本地时间替换原始时间
            task_data['created_at'] = task_data.pop('created_at_local')
            task_data['last_modified'] = task_data.pop('last_modified_local')
            
            # 确保 task_type 字段存在
            if 'task_type' not in task_data:
                task_data['task_type'] = 'main'
            
            # 获取子任务
            task_data['sub_tasks'] = self.get_sub_tasks(task_data['task_id'])
            
            result.append(task_data)
        
        return result
    
    def get_main_task_by_id(self, task_id: str) -> Optional[Dict]:
        """获取指定ID的主任务"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT m.*, ts.last_run_time, ts.next_run_time, ts.last_status,
               ts.total_runs, ts.success_runs, ts.fail_runs,
               ts.avg_duration, ts.last_error, ts.success_rate,
               m.created_at as created_at_local,
               m.last_modified as last_modified_local
        FROM main_tasks m
        LEFT JOIN task_status ts ON m.task_id = ts.task_id
        WHERE m.task_id = ?
        ''', (task_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        task_data = dict(row)
        # 处理JSON字段
        if task_data.get('params'):
            try:
                task_data['params'] = json.loads(task_data['params'])
            except:
                task_data['params'] = {}
        
        if task_data.get('tags'):
            try:
                task_data['tags'] = json.loads(task_data['tags'])
            except:
                task_data['tags'] = []
        
        # 使用本地时间替换原始时间
        task_data['created_at'] = task_data.pop('created_at_local')
        task_data['last_modified'] = task_data.pop('last_modified_local')
        
        return task_data
    
    def create_main_task(self, task_id: str, task_data: Dict) -> bool:
        """创建新的主任务"""
        try:
            cursor = self.conn.cursor()
            
            # 检查任务ID是否已存在
            cursor.execute("SELECT COUNT(*) FROM main_tasks WHERE task_id = ?", (task_id,))
            if cursor.fetchone()[0] > 0:
                logger.error(f"任务ID '{task_id}' 已存在")
                return False
            
            # 准备任务数据
            params = task_data.get('params', {})
            
            # 如果是发送邮件任务，确保params包含必要的内容
            if task_data.get('endpoint') == '/log/send-email' and not params:
                params = {
                    "content": None,
                    "mode": "simple",
                    "subject": "B站历史记录日报 - {current_time}"
                }
                logger.info(f"为发送邮件任务 '{task_id}' 自动添加默认参数")
            
            params_json = json.dumps(params) if params else None
            tags = json.dumps(task_data.get('tags', [])) if task_data.get('tags') else '[]'
            
            # 处理interval类型任务的特殊字段
            interval_value = None
            interval_unit = None
            if task_data.get('schedule_type') == 'interval':
                interval_value = task_data.get('interval_value', task_data.get('interval'))
                interval_unit = task_data.get('interval_unit', task_data.get('unit'))
                logger.info(f"设置间隔执行任务: 每 {interval_value} {interval_unit}")
            
            # 插入主任务
            cursor.execute("""
            INSERT INTO main_tasks (
                task_id, name, endpoint, method, params, schedule_type, 
                schedule_time, schedule_delay, interval_value, interval_unit, 
                enabled, task_type, last_modified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                task_data.get('name', task_id),
                task_data.get('endpoint', ''),
                task_data.get('method', 'GET'),
                params_json,
                task_data.get('schedule_type', 'daily'),
                task_data.get('schedule_time'),
                task_data.get('schedule_delay'),
                interval_value,
                interval_unit,
                task_data.get('enabled', 1),
                'main',
                datetime.now().isoformat()
            ))
            
            # 初始化任务状态
            cursor.execute('''
            INSERT INTO task_status (task_id, tags)
            VALUES (?, ?)
            ''', (task_id, tags))
            
            self.conn.commit()
            logger.info(f"成功创建主任务 '{task_id}'")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"创建主任务失败: {str(e)}")
            return False
    
    def update_main_task(self, task_id: str, task_data: Dict) -> bool:
        """更新主任务信息"""
        try:
            cursor = self.conn.cursor()
            
            # 检查任务是否存在且为主任务
            cursor.execute("SELECT COUNT(*) FROM main_tasks WHERE task_id = ?", (task_id,))
            if cursor.fetchone()[0] == 0:
                logger.error(f"主任务 '{task_id}' 不存在")
                return False
            
            # 准备更新字段
            fields = []
            values = []
            
            for key, value in task_data.items():
                if key in ['name', 'endpoint', 'method', 'schedule_type', 'schedule_time', 
                          'schedule_delay', 'enabled']:
                    fields.append(f"{key} = ?")
                    values.append(value)
                elif key == 'interval_value' or key == 'interval':
                    fields.append("interval_value = ?")
                    values.append(value)
                elif key == 'interval_unit' or key == 'unit':
                    fields.append("interval_unit = ?")
                    values.append(value)
                elif key == 'params':
                    # 如果是发送邮件任务，确保params包含必要的内容
                    if 'endpoint' in task_data and task_data['endpoint'] == '/log/send-email':
                        if not value:
                            value = {
                                "content": None,
                                "mode": "simple",
                                "subject": "B站历史记录日报 - {current_time}"
                            }
                            logger.info(f"为发送邮件任务 '{task_id}' 自动添加默认参数")
                    fields.append("params = ?")
                    values.append(json.dumps(value))
                elif key == 'tags':
                    # 更新任务状态表中的标签
                    try:
                        cursor.execute("""
                        UPDATE task_status
                        SET tags = ?
                        WHERE task_id = ?
                        """, (json.dumps(value), task_id))
                    except Exception as e:
                        logger.error(f"更新任务标签失败: {str(e)}")
            
            # 添加最后修改时间
            fields.append("last_modified = ?")
            values.append(datetime.now().isoformat())
            
            # 添加任务ID
            values.append(task_id)
            
            if fields:
                # 构建更新SQL
                sql = f"UPDATE main_tasks SET {', '.join(fields)} WHERE task_id = ?"
                cursor.execute(sql, values)
                
                self.conn.commit()
                logger.info(f"成功更新主任务 '{task_id}'")
                return True
            else:
                logger.warning(f"没有提供有效的更新字段")
                return False
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"更新主任务失败: {str(e)}")
            return False
    
    def delete_main_task(self, task_id: str) -> bool:
        """删除主任务及其所有子任务"""
        try:
            cursor = self.conn.cursor()
            
            # 检查任务是否存在且为主任务
            cursor.execute("SELECT COUNT(*) FROM main_tasks WHERE task_id = ?", (task_id,))
            if cursor.fetchone()[0] == 0:
                logger.error(f"主任务 '{task_id}' 不存在")
                return False
            
            # 开启事务
            self.conn.execute("BEGIN TRANSACTION")
            
            # 获取所有子任务ID
            cursor.execute("SELECT task_id FROM sub_tasks WHERE parent_id = ?", (task_id,))
            subtask_ids = [row[0] for row in cursor.fetchall()]
            
            # 删除子任务依赖
            for subtask_id in subtask_ids:
                cursor.execute("DELETE FROM task_dependencies WHERE task_id = ? OR depends_on = ?", 
                             (subtask_id, subtask_id))
                             
            # 删除子任务状态
            for subtask_id in subtask_ids:
                cursor.execute("DELETE FROM sub_task_status WHERE task_id = ?", (subtask_id,))
            
            # 删除子任务执行历史记录
            for subtask_id in subtask_ids:
                cursor.execute("DELETE FROM sub_task_executions WHERE task_id = ?", (subtask_id,))
            
            # 删除子任务
            cursor.execute("DELETE FROM sub_tasks WHERE parent_id = ?", (task_id,))
            
            # 删除主任务依赖
            cursor.execute("DELETE FROM task_dependencies WHERE task_id = ? OR depends_on = ?", 
                         (task_id, task_id))
            
            # 删除主任务状态
            cursor.execute("DELETE FROM task_status WHERE task_id = ?", (task_id,))
            
            # 删除主任务执行历史记录
            cursor.execute("DELETE FROM task_executions WHERE task_id = ?", (task_id,))
            
            # 删除主任务
            cursor.execute("DELETE FROM main_tasks WHERE task_id = ?", (task_id,))
            
            self.conn.commit()
            logger.info(f"成功删除主任务 '{task_id}' 及其子任务")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"删除主任务失败: {str(e)}")
            return False
    
    # =================== 子任务管理 ===================
    
    def get_sub_tasks(self, parent_id: str) -> List[Dict]:
        """获取指定主任务的所有子任务"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT s.*, 
               sts.last_run_time, sts.next_run_time, sts.last_status,
               sts.total_runs, sts.success_runs, sts.fail_runs,
               sts.avg_duration, sts.last_error, sts.tags,
               sts.success_rate,
               s.created_at as created_at_local,
               s.last_modified as last_modified_local,
               td.depends_on
        FROM sub_tasks s
        LEFT JOIN sub_task_status sts ON s.task_id = sts.task_id
        LEFT JOIN task_dependencies td ON s.task_id = td.task_id
        WHERE s.parent_id = ?
        ORDER BY s.sequence_number
        ''', (parent_id,))
        
        rows = cursor.fetchall()
        result = []
        
        for row in rows:
            task_data = dict(row)
            # 处理JSON字段
            if task_data.get('params'):
                try:
                    task_data['params'] = json.loads(task_data['params'])
                except:
                    task_data['params'] = {}
            
            if task_data.get('tags'):
                try:
                    task_data['tags'] = json.loads(task_data['tags'])
                except:
                    task_data['tags'] = []
            
            # 使用本地时间替换原始时间
            task_data['created_at'] = task_data.pop('created_at_local')
            task_data['last_modified'] = task_data.pop('last_modified_local')
            
            # 获取依赖任务信息
            cursor.execute('''
            SELECT td.depends_on, COALESCE(mt.name, st.name) as depends_on_name
            FROM task_dependencies td
            LEFT JOIN main_tasks mt ON td.depends_on = mt.task_id
            LEFT JOIN sub_tasks st ON td.depends_on = st.task_id
            WHERE td.task_id = ?
            ''', (task_data['task_id'],))
            
            dependencies = cursor.fetchall()
            if dependencies:
                task_data['depends_on'] = {
                    'task_id': dependencies[0][0],
                    'name': dependencies[0][1]
                }
            
            result.append(task_data)
        
        return result
    
    def get_subtask_by_id(self, task_id: str) -> Optional[Dict]:
        """获取指定ID的子任务"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT s.*, 
               sts.last_run_time, sts.next_run_time, sts.last_status,
               sts.total_runs, sts.success_runs, sts.fail_runs,
               sts.avg_duration, sts.last_error, sts.tags,
               sts.success_rate,
               s.created_at as created_at_local,
               s.last_modified as last_modified_local,
               td.depends_on
        FROM sub_tasks s
        LEFT JOIN sub_task_status sts ON s.task_id = sts.task_id
        LEFT JOIN task_dependencies td ON s.task_id = td.task_id
        WHERE s.task_id = ?
        ''', (task_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        task_data = dict(row)
        # 处理JSON字段
        if task_data.get('params'):
            try:
                task_data['params'] = json.loads(task_data['params'])
            except:
                task_data['params'] = {}
        
        if task_data.get('tags'):
            try:
                task_data['tags'] = json.loads(task_data['tags'])
            except:
                task_data['tags'] = []
        
        # 使用本地时间替换原始时间
        task_data['created_at'] = task_data.pop('created_at_local')
        task_data['last_modified'] = task_data.pop('last_modified_local')
        
        # 获取依赖任务信息
        cursor.execute('''
        SELECT td.depends_on, COALESCE(mt.name, st.name) as depends_on_name
        FROM task_dependencies td
        LEFT JOIN main_tasks mt ON td.depends_on = mt.task_id
        LEFT JOIN sub_tasks st ON td.depends_on = st.task_id
        WHERE td.task_id = ?
        ''', (task_id,))
        
        dependencies = cursor.fetchall()
        if dependencies:
            task_data['depends_on'] = {
                'task_id': dependencies[0][0],
                'name': dependencies[0][1]
            }
        
        return task_data
    
    def get_sub_task(self, parent_id: str, task_id: str) -> Optional[Dict]:
        """获取指定主任务下的特定子任务"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT s.*, 
               sts.last_run_time, sts.next_run_time, sts.last_status,
               sts.total_runs, sts.success_runs, sts.fail_runs,
               sts.avg_duration, sts.last_error, sts.tags,
               sts.success_rate,
               s.created_at as created_at_local,
               s.last_modified as last_modified_local,
               td.depends_on
        FROM sub_tasks s
        LEFT JOIN sub_task_status sts ON s.task_id = sts.task_id
        LEFT JOIN task_dependencies td ON s.task_id = td.task_id
        WHERE s.task_id = ? AND s.parent_id = ?
        ''', (task_id, parent_id))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        task_data = dict(row)
        # 处理JSON字段
        if task_data.get('params'):
            try:
                task_data['params'] = json.loads(task_data['params'])
            except:
                task_data['params'] = {}
        
        if task_data.get('tags'):
            try:
                task_data['tags'] = json.loads(task_data['tags'])
            except:
                task_data['tags'] = []
        
        # 使用本地时间替换原始时间
        task_data['created_at'] = task_data.pop('created_at_local')
        task_data['last_modified'] = task_data.pop('last_modified_local')
        
        # 获取依赖任务信息
        cursor.execute('''
        SELECT td.depends_on, COALESCE(mt.name, st.name) as depends_on_name
        FROM task_dependencies td
        LEFT JOIN main_tasks mt ON td.depends_on = mt.task_id
        LEFT JOIN sub_tasks st ON td.depends_on = st.task_id
        WHERE td.task_id = ?
        ''', (task_id,))
        
        dependencies = cursor.fetchall()
        if dependencies:
            task_data['depends_on'] = {
                'task_id': dependencies[0][0],
                'name': dependencies[0][1]
            }
        
        return task_data
    
    def create_sub_task(self, parent_id: str, task_data: Dict) -> bool:
        """创建新的子任务"""
        try:
            cursor = self.conn.cursor()
            logger.info(f"开始创建子任务，父任务ID: {parent_id}")
            logger.info(f"任务数据: {json.dumps(task_data, ensure_ascii=False)}")
            
            # 检查父任务是否存在
            cursor.execute("SELECT COUNT(*) FROM main_tasks WHERE task_id = ?", (parent_id,))
            if cursor.fetchone()[0] == 0:
                logger.error(f"父任务 '{parent_id}' 不存在")
                return False
            
            # 获取当前最大序号
            cursor.execute("""
            SELECT COALESCE(MAX(sequence_number), 0)
            FROM sub_tasks
            WHERE parent_id = ?
            """, (parent_id,))
            max_sequence = cursor.fetchone()[0]
            logger.info(f"当前最大序号: {max_sequence}")
            
            # 准备任务数据
            task_id = task_data.get('task_id')
            if not task_id:
                logger.error("未提供子任务ID")
                return False
                
            # 检查任务ID是否已存在
            cursor.execute("""
            SELECT COUNT(*) FROM sub_tasks WHERE task_id = ?
            """, (task_id,))
            if cursor.fetchone()[0] > 0:
                logger.error(f"子任务ID '{task_id}' 已存在")
                return False
            
            params = task_data.get('params', {})
            
            # 如果是发送邮件任务，确保params包含必要的内容
            if task_data.get('endpoint') == '/log/send-email' and not params:
                params = {
                    "content": None,
                    "mode": "simple",
                    "subject": "B站历史记录日报 - {current_time}"
                }
                logger.info(f"为发送邮件子任务 '{task_id}' 自动添加默认参数")
            
            params_json = json.dumps(params) if params else None
            tags = json.dumps(task_data.get('tags', [])) if task_data.get('tags') else '[]'
            
            # 插入子任务
            logger.info(f"开始插入子任务记录: {task_id}")
            cursor.execute("""
            INSERT INTO sub_tasks (
                task_id, parent_id, name, sequence_number,
                endpoint, method, params, schedule_type, enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                parent_id,
                task_data.get('name', task_id),
                max_sequence + 1,
                task_data.get('endpoint', ''),
                task_data.get('method', 'GET'),
                params_json,
                task_data.get('schedule_type', 'daily'),
                task_data.get('enabled', 1)
            ))
            logger.info("子任务记录插入成功")
            
            # 初始化子任务状态
            logger.info(f"开始初始化子任务状态: {task_id}")
            cursor.execute('''
            INSERT INTO sub_task_status (task_id, tags)
            VALUES (?, ?)
            ''', (task_id, tags))
            logger.info("子任务状态初始化成功")
            
            # 处理依赖关系
            if 'depends_on' in task_data:
                logger.info(f"发现依赖关系配置: {json.dumps(task_data.get('depends_on', {}), ensure_ascii=False)}")
                if task_data['depends_on'] and isinstance(task_data['depends_on'], dict):
                    depends_on_id = task_data['depends_on'].get('task_id')
                    logger.info(f"依赖任务ID: {depends_on_id}")
                    if depends_on_id:
                        try:
                            logger.info(f"开始插入依赖关系: {task_id} -> {depends_on_id}")
                            cursor.execute("""
                            INSERT INTO task_dependencies (task_id, depends_on)
                            VALUES (?, ?)
                            """, (task_id, depends_on_id))
                            logger.info("依赖关系插入成功")
                        except Exception as e:
                            logger.error(f"插入依赖关系时出错: {str(e)}")
                            raise
                else:
                    logger.warning("依赖关系数据格式不正确")
            else:
                logger.info("没有依赖关系需要处理")
            
            self.conn.commit()
            logger.info(f"成功创建子任务 '{task_id}'")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"创建子任务失败: {str(e)}")
            return False
    
    def update_subtask(self, task_id: str, task_data: Dict) -> bool:
        """更新子任务信息"""
        try:
            cursor = self.conn.cursor()
            
            # 检查任务是否存在且为子任务
            cursor.execute("SELECT COUNT(*) FROM sub_tasks WHERE task_id = ?", (task_id,))
            if cursor.fetchone()[0] == 0:
                logger.error(f"子任务 '{task_id}' 不存在")
                return False
            
            # 准备更新字段
            fields = []
            values = []
            
            for key, value in task_data.items():
                if key in ['name', 'endpoint', 'method', 'schedule_type', 'enabled']:
                    fields.append(f"{key} = ?")
                    values.append(value)
                elif key == 'params':
                    # 如果是发送邮件任务，确保params包含必要的内容
                    if 'endpoint' in task_data and task_data['endpoint'] == '/log/send-email':
                        if not value:
                            value = {
                                "content": None,
                                "mode": "simple",
                                "subject": "B站历史记录日报 - {current_time}"
                            }
                            logger.info(f"为发送邮件子任务 '{task_id}' 自动添加默认参数")
                    fields.append("params = ?")
                    values.append(json.dumps(value))
                elif key == 'tags':
                    # 更新子任务状态表中的标签
                    try:
                        cursor.execute("""
                        UPDATE sub_task_status
                        SET tags = ?
                        WHERE task_id = ?
                        """, (json.dumps(value), task_id))
                    except Exception as e:
                        logger.error(f"更新子任务标签失败: {str(e)}")
            
            # 添加最后修改时间
            fields.append("last_modified = ?")
            values.append(datetime.now().isoformat())
            
            # 更新依赖关系
            if 'depends_on' in task_data:
                # 删除现有依赖
                cursor.execute("DELETE FROM task_dependencies WHERE task_id = ?", (task_id,))
                
                # 添加新依赖
                if task_data['depends_on'] and isinstance(task_data['depends_on'], dict):
                    depends_on_id = task_data['depends_on'].get('task_id')
                    if depends_on_id:
                        try:
                            cursor.execute("""
                            INSERT INTO task_dependencies (task_id, depends_on)
                            VALUES (?, ?)
                            """, (task_id, depends_on_id))
                        except Exception as e:
                            logger.error(f"更新依赖关系失败: {str(e)}")
            
            # 添加任务ID
            values.append(task_id)
            
            if fields:
                # 构建更新SQL
                sql = f"UPDATE sub_tasks SET {', '.join(fields)} WHERE task_id = ?"
                cursor.execute(sql, values)
                
                self.conn.commit()
                logger.info(f"成功更新子任务 '{task_id}'")
                return True
            else:
                logger.warning(f"没有提供有效的更新字段")
                return False
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"更新子任务失败: {str(e)}")
            return False
    
    def delete_subtask(self, task_id: str, parent_id: str = None) -> bool:
        """删除子任务，可选指定父任务ID以确保只删除特定主任务下的子任务"""
        try:
            cursor = self.conn.cursor()
            
            # 检查任务是否存在且为子任务
            if parent_id:
                cursor.execute("SELECT COUNT(*) FROM sub_tasks WHERE task_id = ? AND parent_id = ?", (task_id, parent_id))
            else:
                cursor.execute("SELECT COUNT(*) FROM sub_tasks WHERE task_id = ?", (task_id,))
                
            if cursor.fetchone()[0] == 0:
                logger.error(f"子任务 '{task_id}' 不存在" + (f" 或不属于主任务 '{parent_id}'" if parent_id else ""))
                return False
            
            # 开启事务
            self.conn.execute("BEGIN TRANSACTION")
            
            # 删除依赖关系
            cursor.execute("DELETE FROM task_dependencies WHERE task_id = ? OR depends_on = ?", 
                         (task_id, task_id))
            
            # 删除子任务状态
            cursor.execute("DELETE FROM sub_task_status WHERE task_id = ?", (task_id,))
            
            # 删除子任务执行历史记录
            cursor.execute("DELETE FROM sub_task_executions WHERE task_id = ?", (task_id,))
            
            # 删除子任务
            if parent_id:
                cursor.execute("DELETE FROM sub_tasks WHERE task_id = ? AND parent_id = ?", (task_id, parent_id))
            else:
                cursor.execute("DELETE FROM sub_tasks WHERE task_id = ?", (task_id,))
            
            # 重新排序剩余子任务
            # 如果提供了parent_id，只重新排序该主任务下的子任务
            if parent_id:
                cursor.execute("""
                SELECT task_id, sequence_number 
                FROM sub_tasks 
                WHERE parent_id = ?
                ORDER BY sequence_number
                """, (parent_id,))
                subtasks = cursor.fetchall()
                
                # 更新序号
                for i, task in enumerate(subtasks, 1):
                    if task[1] != i:  # 如果序号不匹配才更新
                        cursor.execute("""
                        UPDATE sub_tasks SET sequence_number = ? WHERE task_id = ?
                        """, (i, task[0]))
            else:
                # 按父任务分组重新排序所有子任务
                cursor.execute("""
                SELECT task_id, parent_id, sequence_number 
                FROM sub_tasks 
                ORDER BY parent_id, sequence_number
                """)
                subtasks = cursor.fetchall()
                
                # 按父任务分组
                subtasks_by_parent = {}
                for subtask in subtasks:
                    if subtask[1] not in subtasks_by_parent:
                        subtasks_by_parent[subtask[1]] = []
                    subtasks_by_parent[subtask[1]].append(subtask)
                
                # 更新序号
                for parent_id, tasks in subtasks_by_parent.items():
                    for i, task in enumerate(tasks, 1):
                        if task[2] != i:  # 如果序号不匹配才更新
                            cursor.execute("""
                            UPDATE sub_tasks SET sequence_number = ? WHERE task_id = ?
                            """, (i, task[0]))
            
            self.conn.commit()
            logger.info(f"成功删除子任务 '{task_id}'" + (f" (主任务: {parent_id})" if parent_id else ""))
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"删除子任务失败: {str(e)}")
            return False
    
    def delete_sub_task(self, parent_id: str, task_id: str) -> bool:
        """删除指定主任务下的特定子任务，确保ID相同时不会错删除主任务"""
        return self.delete_subtask(task_id, parent_id)
    
    def reorder_subtasks(self, parent_id: str, task_order: List[str]) -> bool:
        """重新排序子任务"""
        try:
            cursor = self.conn.cursor()
            
            # 检查主任务是否存在
            cursor.execute("SELECT COUNT(*) FROM main_tasks WHERE task_id = ?", (parent_id,))
            if cursor.fetchone()[0] == 0:
                logger.error(f"主任务 '{parent_id}' 不存在")
                return False
            
            # 获取所有子任务
            cursor.execute("""
            SELECT task_id FROM sub_tasks WHERE parent_id = ?
            """, (parent_id,))
            existing_subtasks = [row[0] for row in cursor.fetchall()]
            
            # 验证输入的任务列表
            if set(task_order) != set(existing_subtasks):
                logger.error(f"提供的任务列表与实际子任务不匹配")
                return False
            
            # 更新序号
            for i, task_id in enumerate(task_order, 1):
                cursor.execute("""
                UPDATE sub_tasks SET sequence_number = ? WHERE task_id = ?
                """, (i, task_id))
            
            self.conn.commit()
            logger.info(f"成功重排主任务 '{parent_id}' 的子任务顺序")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"重排子任务顺序失败: {str(e)}")
            return False
    
    def is_main_task(self, task_id: str) -> bool:
        """检查指定任务是否为主任务"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM main_tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        return row and row[0] > 0
    
    def get_task_execution_history_enhanced(
        self,
        task_id: str,
        include_subtasks: bool = True,
        conditions: dict = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """获取任务的执行历史（增强版）"""
        cursor = self.conn.cursor()
        params = []
        where_clauses = []
        
        # 基础查询
        if self.is_main_task(task_id):
            # 主任务查询
            base_query = """
            SELECT te.*, mt.name as task_name, 'main' as task_type, NULL as parent_id
            FROM task_executions te
            LEFT JOIN main_tasks mt ON te.task_id = mt.task_id
            """
            where_clauses.append("te.task_id = ?")
            params.append(task_id)
            
            if include_subtasks:
                # 添加子任务历史
                base_query = """
                SELECT te.*, mt.name as task_name, 'main' as task_type, NULL as parent_id
                FROM task_executions te
                LEFT JOIN main_tasks mt ON te.task_id = mt.task_id
                WHERE te.task_id = ?
            UNION ALL
                SELECT ste.*, st.name as task_name, 'sub' as task_type, st.parent_id
                FROM sub_task_executions ste
                LEFT JOIN sub_tasks st ON ste.task_id = st.task_id
                WHERE st.parent_id = ?
                """
                params = [task_id, task_id]  # 重置参数列表
        else:
            # 子任务查询
            base_query = """
            SELECT ste.*, st.name as task_name, 'sub' as task_type, st.parent_id
            FROM sub_task_executions ste
            LEFT JOIN sub_tasks st ON ste.task_id = st.task_id
            """
            where_clauses.append("ste.task_id = ?")
            params.append(task_id)
        
        # 添加条件过滤
        if conditions:
            if conditions.get('status'):
                where_clauses.append("status = ?")
                params.append(conditions['status'])
            if conditions.get('start_date'):
                where_clauses.append("start_time >= ?")
                params.append(conditions['start_date'])
            if conditions.get('end_date'):
                where_clauses.append("start_time <= ?")
                params.append(conditions['end_date'])
        
        # 构建完整查询
        if not include_subtasks or not self.is_main_task(task_id):
            # 如果不包含子任务或者是子任务查询，使用 WHERE 子句
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            final_query = f"{base_query} WHERE {where_clause}"
        else:
            # 如果是包含子任务的主任务查询，base_query 已经包含了完整的查询条件
            final_query = base_query
        
        # 获取总记录数
        count_query = f"SELECT COUNT(*) FROM ({final_query})"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        # 添加分页和排序
        final_query = f"""
        {final_query}
            ORDER BY start_time DESC
        LIMIT ? OFFSET ?
        """
        params.extend([page_size, (page - 1) * page_size])
        
        # 执行查询
        cursor.execute(final_query, params)
        rows = cursor.fetchall()
        
        # 处理结果
        records = []
        for row in rows:
            record = dict(row)
            # 处理输出字段（如果存在）
            if record.get('output'):
                try:
                    record['output'] = json.loads(record['output'])
                except:
                    pass
            records.append(record)
        
        return {
            'records': records,
            'total': total_count
        }

    def get_task_dependencies(self, task_id: str) -> List[str]:
        """获取任务的依赖项"""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT depends_on FROM task_dependencies WHERE task_id = ?
        """, (task_id,))
        return [row[0] for row in cursor.fetchall()]

    def record_task_execution_enhanced(self, 
                                     task_id: str, 
                                     start_time: str,
                                     end_time: str = None,
                                     duration: float = None,
                                     status: str = "success",
                                     error_message: str = None,
                                     triggered_by: str = None,
                                     output: str = None,
                                     parent_execution_id: int = None,
                                     next_run_time: str = None) -> int:
        """记录任务执行（增强版）"""
        try:
            cursor = self.conn.cursor()
            
            # 如果提供了开始时间和结束时间，但没有提供持续时间，尝试计算
            if start_time and end_time and duration is None:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    duration = (end_dt - start_dt).total_seconds()
                except Exception as e:
                    logger.warning(f"计算任务持续时间失败: {str(e)}")
            
            # 确定任务类型（主任务或子任务）
            is_sub_task = not self.is_main_task(task_id)
            
            # 获取任务配置以确定下次执行时间
            task_config = None
            if is_sub_task:
                task_config = self.get_subtask_by_id(task_id)
            else:
                task_config = self.get_main_task_by_id(task_id)
            
            # 如果没有提供 next_run_time，则计算它
            if next_run_time is None:
                schedule_type = task_config.get('schedule_type') if task_config else None
                
                # 只有主任务时才计算下次执行时间
                if not is_sub_task:
                    if schedule_type == 'daily':
                        schedule_time = task_config.get('schedule_time')
                        if schedule_time:
                            try:
                                current_dt = datetime.fromisoformat(start_time)
                                schedule_parts = schedule_time.split(':')
                                next_dt = current_dt.replace(
                                    hour=int(schedule_parts[0]),
                                    minute=int(schedule_parts[1]),
                                    second=0,
                                    microsecond=0
                                )
                                
                                # 如果当前时间已经过了今天的调度时间，设置为明天
                                if current_dt >= next_dt:
                                    next_dt = next_dt + timedelta(days=1)
                                
                                next_run_time = next_dt.strftime('%Y-%m-%d %H:%M:%S')
                                logger.info(f"计算得到下次执行时间: {next_run_time}")
                            except Exception as e:
                                logger.error(f"计算下次执行时间失败: {str(e)}")
                                next_run_time = None
                    elif schedule_type == 'interval':
                        # 处理间隔任务
                        interval_value = task_config.get('interval_value')
                        interval_unit = task_config.get('interval_unit')
                        
                        if interval_value and interval_unit:
                            try:
                                current_dt = datetime.fromisoformat(start_time if end_time is None else end_time)
                                
                                # 根据间隔值和单位计算下次执行时间
                                if interval_unit == 'minutes':
                                    next_dt = current_dt + timedelta(minutes=interval_value)
                                elif interval_unit == 'hours':
                                    next_dt = current_dt + timedelta(hours=interval_value)
                                elif interval_unit == 'days':
                                    next_dt = current_dt + timedelta(days=interval_value)
                                elif interval_unit == 'weeks':
                                    next_dt = current_dt + timedelta(weeks=interval_value)
                                elif interval_unit == 'months':
                                    # 手动计算月份
                                    year = current_dt.year
                                    month = current_dt.month + interval_value
                                    
                                    # 处理月份溢出
                                    while month > 12:
                                        month -= 12
                                        year += 1
                                    
                                    # 处理月份天数问题（例如，1月31日 + 1个月）
                                    day = min(current_dt.day, calendar.monthrange(year, month)[1])
                                    
                                    next_dt = current_dt.replace(year=year, month=month, day=day)
                                elif interval_unit == 'years':
                                    # 处理闰年问题
                                    year = current_dt.year + interval_value
                                    month = current_dt.month
                                    day = min(current_dt.day, calendar.monthrange(year, month)[1])
                                    
                                    next_dt = current_dt.replace(year=year, day=day)
                                else:
                                    logger.warning(f"不支持的间隔单位: {interval_unit}")
                                    next_dt = None
                                
                                if next_dt:
                                    next_run_time = next_dt.strftime('%Y-%m-%d %H:%M:%S')
                                    logger.info(f"计算得到间隔任务下次执行时间: {next_run_time}，间隔: {interval_value} {interval_unit}")
                            except Exception as e:
                                logger.error(f"计算间隔任务下次执行时间失败: {str(e)}")
                                next_run_time = None
            
            # 根据任务类型选择表
            table_name = "sub_task_executions" if is_sub_task else "task_executions"
            status_table = "sub_task_status" if is_sub_task else "task_status"
            
            # 构建插入语句
            fields = ["task_id", "start_time", "end_time", "duration", 
                     "status", "error_message", "triggered_by", "output"]
            values = [task_id, start_time, end_time, duration,
                     status, error_message, triggered_by, output]
            
            if next_run_time is not None:
                fields.append("next_run_time")
                values.append(next_run_time)
            
            # 如果提供了父执行ID，添加到字段列表
            if parent_execution_id is not None:
                fields.append("parent_execution_id")
                values.append(parent_execution_id)
            
            # 构建SQL语句
            placeholders = ["?" for _ in values]
            sql = f"""
            INSERT INTO {table_name} 
            ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            """
            
            # 执行插入
            cursor.execute(sql, values)
            execution_id = cursor.lastrowid
            
            # 更新任务状态
            # 先获取当前状态以计算成功率和平均执行时间
            cursor.execute(f'''
            SELECT total_runs, success_runs, fail_runs, avg_duration
            FROM {status_table}
            WHERE task_id = ?
            ''', (task_id,))
            current_stats = cursor.fetchone()
            
            if current_stats:
                total_runs = current_stats[0] + 1
                success_runs = current_stats[1] + (1 if status == 'success' else 0)
                fail_runs = current_stats[2] + (1 if status != 'success' else 0)
                success_rate = (success_runs / total_runs) * 100 if total_runs > 0 else 0
                
                # 计算新的平均执行时间
                current_avg_duration = current_stats[3] or 0
                if duration is not None:
                    avg_duration = ((current_avg_duration * (total_runs - 1)) + duration) / total_runs
                else:
                    avg_duration = current_avg_duration
                
                cursor.execute(f'''
                UPDATE {status_table}
                SET last_run_time = ?,
                    next_run_time = ?,
                    last_status = ?,
                    last_error = ?,
                    total_runs = ?,
                    success_runs = ?,
                    fail_runs = ?,
                    success_rate = ?,
                    avg_duration = ?
                WHERE task_id = ?
                ''', (
                    start_time,
                    next_run_time,
                    status,
                    error_message,
                    total_runs,
                    success_runs,
                    fail_runs,
                    success_rate,
                    avg_duration,
                    task_id
                ))
            else:
                # 如果没有状态记录，创建一个新的
                cursor.execute(f'''
                INSERT INTO {status_table}
                (task_id, last_run_time, next_run_time, last_status, last_error,
                 total_runs, success_runs, fail_runs, success_rate, avg_duration)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ''', (
                    task_id,
                    start_time,
                    next_run_time,
                    status,
                    error_message,
                    1 if status == 'success' else 0,
                    1 if status != 'success' else 0,
                    100 if status == 'success' else 0,
                    duration or 0
                ))
            
            self.conn.commit()
            return execution_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"记录任务执行失败: {str(e)}")
            return -1

    def _calculate_next_run_time(self, task_data: Dict) -> Optional[datetime]:
        """计算下次执行时间"""
        schedule_type = task_data.get('schedule_type')
        if not schedule_type:
            return None
            
        now = datetime.now()
        
        if schedule_type == 'daily':
            schedule_time = task_data.get('schedule_time')
            if not schedule_time:
                return None
            hour, minute = map(int, schedule_time.split(':'))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
                
        elif schedule_type == 'interval':
            interval = task_data.get('interval', 1)
            unit = task_data.get('unit', 'hours')
            
            if unit == 'minutes':
                next_run = now + timedelta(minutes=interval)
            elif unit == 'hours':
                next_run = now + timedelta(hours=interval)
            elif unit == 'days':
                next_run = now + timedelta(days=interval)
            elif unit == 'months':
                next_run = now + relativedelta(months=interval)
            elif unit == 'years':
                next_run = now + relativedelta(years=interval)
            else:
                return None
                
        elif schedule_type == 'once':
            delay = task_data.get('delay', 0)
            next_run = now + timedelta(seconds=delay)
            
        else:
            return None
            
        return next_run

    def update_next_execution_time(self, task_id: str, next_run_time: Optional[str] = None):
        """
        更新任务的下次执行时间
        
        Args:
            task_id: 任务ID
            next_run_time: 下次执行时间，如果为None则自动计算
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 确定任务类型（主任务或子任务）
            is_sub_task = not self.is_main_task(task_id)
            
            # 如果是子任务，不处理下次执行时间
            if is_sub_task:
                logger.warning(f"子任务 {task_id} 不支持更新下次执行时间")
                return False
            
            # 获取任务配置
            task_config = self.get_main_task_by_id(task_id)
            if not task_config:
                logger.error(f"任务 {task_id} 不存在")
                return False
            
            # 如果没有提供 next_run_time，则计算它
            if next_run_time is None:
                schedule_type = task_config.get('schedule_type')
                
                if schedule_type == 'daily':
                    schedule_time = task_config.get('schedule_time')
                    if schedule_time:
                        try:
                            now = datetime.now()
                            schedule_parts = schedule_time.split(':')
                            next_dt = now.replace(
                                hour=int(schedule_parts[0]),
                                minute=int(schedule_parts[1]),
                                second=0,
                                microsecond=0
                            )
                            
                            # 如果当前时间已经过了今天的调度时间，设置为明天
                            if now >= next_dt:
                                next_dt = next_dt + timedelta(days=1)
                            
                            next_run_time = next_dt.strftime('%Y-%m-%d %H:%M:%S')
                            logger.info(f"计算得到下次执行时间: {next_run_time}")
                        except Exception as e:
                            logger.error(f"计算下次执行时间失败: {str(e)}")
                            return False
                
                elif schedule_type == 'interval':
                    # 处理间隔任务
                    interval_value = task_config.get('interval_value')
                    interval_unit = task_config.get('interval_unit')
                    
                    if interval_value and interval_unit:
                        try:
                            now = datetime.now()
                            
                            # 根据间隔值和单位计算下次执行时间
                            if interval_unit == 'minutes':
                                next_dt = now + timedelta(minutes=interval_value)
                            elif interval_unit == 'hours':
                                next_dt = now + timedelta(hours=interval_value)
                            elif interval_unit == 'days':
                                next_dt = now + timedelta(days=interval_value)
                            elif interval_unit == 'weeks':
                                next_dt = now + timedelta(weeks=interval_value)
                            elif interval_unit == 'months':
                                # 手动计算月份
                                year = now.year
                                month = now.month + interval_value
                                
                                # 处理月份溢出
                                while month > 12:
                                    month -= 12
                                    year += 1
                                
                                # 处理月份天数问题
                                day = min(now.day, calendar.monthrange(year, month)[1])
                                
                                next_dt = now.replace(year=year, month=month, day=day)
                            elif interval_unit == 'years':
                                # 处理闰年问题
                                year = now.year + interval_value
                                month = now.month
                                day = min(now.day, calendar.monthrange(year, month)[1])
                                
                                next_dt = now.replace(year=year, day=day)
                            else:
                                logger.warning(f"不支持的间隔单位: {interval_unit}")
                                return False
                            
                            next_run_time = next_dt.strftime('%Y-%m-%d %H:%M:%S')
                            logger.info(f"计算得到间隔任务下次执行时间: {next_run_time}，间隔: {interval_value} {interval_unit}")
                        except Exception as e:
                            logger.error(f"计算间隔任务下次执行时间失败: {str(e)}")
                            return False
                else:
                    logger.warning(f"任务 {task_id} 的计划类型 {schedule_type} 不支持自动计算下次执行时间")
                    return False
            
            # 更新数据库中的下次执行时间
            cursor = self.conn.cursor()
            cursor.execute('''
            UPDATE task_status
            SET next_run_time = ?
            WHERE task_id = ?
            ''', (next_run_time, task_id))
            
            # 如果没有状态记录，创建一个新的
            if cursor.rowcount == 0:
                cursor.execute('''
                INSERT INTO task_status
                (task_id, next_run_time)
                VALUES (?, ?)
                ''', (task_id, next_run_time))
            
            self.conn.commit()
            logger.info(f"成功更新任务 {task_id} 的下次执行时间: {next_run_time}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"更新任务下次执行时间失败: {str(e)}")
            return False

    def load_config(self):
        """加载调度器配置"""
        try:
            from scripts.utils import get_config_path
            config_path = get_config_path('scheduler_config.yaml')
            
            if not os.path.exists(config_path):
                return False
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if 'tasks' not in config:
                return False
                
            self.config = config
            return True
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False