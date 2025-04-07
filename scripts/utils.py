import logging
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, Any

import yaml


def get_base_path() -> str:
    """获取项目基础路径"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的exe运行，返回exe所在目录
        return os.path.dirname(sys.executable)
    else:
        # 如果是直接运行python脚本
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_config_path(config_file: str) -> str:
    """
    获取配置文件路径
    Args:
        config_file: 配置文件名
    Returns:
        配置文件的完整路径
    """
    if getattr(sys, 'frozen', False):
        # 如果是打包后的exe运行，配置文件在_internal/config目录中
        base_path = os.path.dirname(sys.executable)
        return os.path.join(base_path, '_internal', 'config', config_file)
    else:
        # 如果是直接运行python脚本
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, 'config', config_file)

def load_config() -> Dict[str, Any]:
    """加载配置文件并验证"""
    try:
        config_path = get_config_path('config.yaml')
        if not os.path.exists(config_path):
            # 打印更多调试信息
            base_path = get_base_path()
            print(f"\n=== 配置文件信息 ===")
            print(f"当前基础路径: {base_path}")
            print(f"尝试加载配置文件: {config_path}")
            print(f"当前目录内容: {os.listdir(base_path)}")
            if os.path.exists(os.path.dirname(config_path)):
                print(f"配置目录内容: {os.listdir(os.path.dirname(config_path))}")
            print("=====================\n")
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证邮件配置
        email_config = config.get('email', {})
        required_fields = ['smtp_server', 'smtp_port', 'sender', 'password', 'receiver']
        missing_fields = [field for field in required_fields if not email_config.get(field)]
        
        if missing_fields:
            raise ValueError(f"邮件配置缺少必要字段: {', '.join(missing_fields)}")
        
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise

def get_output_path(*paths: str) -> str:
    """
    获取输出文件路径
    Args:
        *paths: 路径片段
    Returns:
        完整的输出路径
    """
    # 总是使用exe所在目录（或项目根目录）作为基础路径
    base_path = get_base_path()
    
    # 基础输出目录
    output_dir = os.path.join(base_path, 'output')
    
    # 创建基础输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 组合完整路径
    full_path = os.path.join(output_dir, *paths)
    
    # 确保父目录存在
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    return full_path

def get_logs_path() -> str:
    """获取日志文件路径"""
    current_time = datetime.now()
    log_path = get_output_path(
        'logs',
        str(current_time.year),
        f"{current_time.month:02d}",
        f"{current_time.day:02d}.log"
    )
    return log_path

def get_db():
    """获取数据库连接"""
    db_path = get_output_path('bilibili_history.db')
    return sqlite3.connect(db_path)
