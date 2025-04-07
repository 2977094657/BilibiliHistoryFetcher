import logging
import time
from datetime import datetime
from pathlib import Path

import requests
import schedule
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class TaskScheduler:
    def __init__(self, config_path=None):
        # 设置配置文件路径
        if config_path is None:
            from scripts.utils import get_config_path
            config_path = get_config_path('scheduler_config.yaml')
            
        self.config_path = config_path
        self.load_config()
        self.task_chains = {}
        self._init_task_status()
        self.base_url = "http://localhost:8899"
        self.tasks = {}
        self.start_time = datetime.now()
        logging.info(f"调度器初始化完成，启动时间: {self.start_time}")

    def load_config(self):
        """加载调度配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.base_url = config.get('base_url', self.base_url)
                self.tasks = config.get('tasks', {})
            logging.info(f"成功加载配置文件: {self.config_path}")
            logging.info(f"已配置的任务: {list(self.tasks.keys())}")
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")

    def execute_task(self, task_name):
        """执行指定任务"""
        if task_name not in self.tasks:
            logging.error(f"未找到任务: {task_name}")
            return

        task = self.tasks[task_name]
        url = f"{self.base_url}{task['endpoint']}"
        method = task.get('method', 'GET').upper()
        params = task.get('params', {})
        
        try:
            logging.info(f"开始执行任务: {task_name}")
            logging.info(f"请求: {method} {url}")
            logging.info(f"参数: {params}")
            
            if method == 'GET':
                response = requests.get(url, params=params)
            elif method == 'POST':
                response = requests.post(url, json=params)
            else:
                logging.error(f"不支持的HTTP方法: {method}")
                return

            if response.status_code == 200:
                logging.info(f"任务 {task_name} 执行成功")
                logging.info(f"响应: {response.json()}")
            else:
                logging.error(f"任务 {task_name} 执行失败: {response.status_code}")
                logging.error(f"错误信息: {response.text}")

        except Exception as e:
            logging.error(f"执行任务 {task_name} 时发生错误: {e}")

    def schedule_tasks(self):
        """设置所有任务的调度"""
        for task_name, task in self.tasks.items():
            if 'schedule' in task:
                schedule_info = task['schedule']
                schedule_type = schedule_info.get('type')
                
                if schedule_type == 'daily':
                    time_str = schedule_info.get('time', '00:00')
                    schedule.every().day.at(time_str).do(self.execute_task, task_name)
                    logging.info(f"已设置每日 {time_str} 执行任务: {task_name}")
                
                elif schedule_type == 'interval':
                    interval = schedule_info.get('interval', 1)
                    unit = schedule_info.get('unit', 'hours')
                    
                    if unit == 'minutes':
                        schedule.every(interval).minutes.do(self.execute_task, task_name)
                    elif unit == 'hours':
                        schedule.every(interval).hours.do(self.execute_task, task_name)
                    elif unit == 'days':
                        schedule.every(interval).days.do(self.execute_task, task_name)
                    elif unit == 'months':
                        schedule.every(interval).months.do(self.execute_task, task_name)
                    elif unit == 'years':
                        schedule.every(interval).years.do(self.execute_task, task_name)
                    
                    logging.info(f"已设置每 {interval} {unit} 执行任务: {task_name}")

    def run(self):
        """运行调度器"""
        self.schedule_tasks()
        logging.info("调度器已启动，等待执行任务...")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"调度器运行错误: {e}")
                time.sleep(60)  # 发生错误时等待1分钟后继续

def create_default_config():
    """创建默认配置文件"""
    default_config = {
        'base_url': 'http://localhost:8000',
        'tasks': {
            'fetch_history': {
                'name': '获取历史记录',
                'endpoint': '/fetch/bili-history',
                'method': 'GET',
                'params': {},
                'schedule': {
                    'type': 'daily',
                    'time': '00:00'
                }
            },
            'clean_data': {
                'name': '清理数据',
                'endpoint': '/clean/data',
                'method': 'POST',
                'params': {},
                'schedule': {
                    'type': 'interval',
                    'interval': 12,
                    'unit': 'hours'
                }
            }
        }
    }

    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / 'scheduler_config.yaml'
    
    if not config_path.exists():
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True, sort_keys=False)
        logging.info(f"已创建默认配置文件: {config_path}")
    
    return config_path

if __name__ == '__main__':
    # 确保配置文件存在
    config_path = create_default_config()
    
    # 创建并运行调度器
    scheduler = TaskScheduler(config_path)
    scheduler.run() 