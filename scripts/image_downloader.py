import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import time
import hashlib
import random
import threading
import sqlite3
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.utils import get_output_path, load_config

config = load_config()

class DownloadStatusDB:
    def __init__(self):
        self.db_path = get_output_path('image_downloads.db')
        self.lock = threading.Lock()
        self._init_db()
    
    def _get_table_name(self, type: str, year: int) -> str:
        """获取表名
        
        Args:
            type: 图片类型 (cover 或 avatar)
            year: 年份
            
        Returns:
            str: 表名，格式为 images_{type}_{year}
        """
        return f"images_{type}s_{year}"
    
    def _create_table(self, cursor: sqlite3.Cursor, table_name: str):
        """创建表结构"""
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                path TEXT NOT NULL,
                downloaded BOOLEAN DEFAULT 0,
                timestamp INTEGER NOT NULL,
                error TEXT
            )
        """)
        
        # 创建索引
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_downloaded 
            ON {table_name}(downloaded)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
            ON {table_name}(timestamp)
        """)
    
    def _init_db(self):
        """初始化数据库表结构"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # 获取现有的表
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'images_%'
                """)
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                # 如果存在旧的单表，迁移数据
                if 'image_downloads' in existing_tables:
                    print("检测到旧的单表结构，开始迁移数据...")
                    self._migrate_from_single_table(cursor)
                    cursor.execute("DROP TABLE image_downloads")
                    print("旧表迁移完成并删除")
                
                conn.commit()
            finally:
                conn.close()
    
    def _migrate_from_single_table(self, cursor: sqlite3.Cursor):
        """从旧的单表结构迁移数据到新的分表结构"""
        # 获取所有数据
        cursor.execute("""
            SELECT hash, type, url, path, downloaded, timestamp, error, year
            FROM image_downloads
        """)
        rows = cursor.fetchall()
        
        # 按年份和类型分组数据
        data_by_table = {}
        for row in rows:
            hash_value, type_name, url, path, downloaded, timestamp, error, year = row
            if not year:  # 如果没有年份信息，跳过
                continue
                
            table_name = self._get_table_name(type_name, year)
            if table_name not in data_by_table:
                data_by_table[table_name] = []
            data_by_table[table_name].append(
                (hash_value, url, path, downloaded, timestamp, error)
            )
        
        # 创建新表并插入数据
        for table_name, table_data in data_by_table.items():
            self._create_table(cursor, table_name)
            cursor.executemany(f"""
                INSERT OR REPLACE INTO {table_name}
                (hash, url, path, downloaded, timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, table_data)
    
    def update_status(self, hash_value: str, type: str, url: str, path: str, 
                     downloaded: bool, error: str = None, year: int = None):
        """更新下载状态"""
        if not year:
            print(f"警告：缺少年份信息，无法更新状态")
            return
            
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                table_name = self._get_table_name(type, year)
                
                # 确保表存在
                self._create_table(cursor, table_name)
                
                # 更新状态
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (hash, url, path, downloaded, timestamp, error)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (hash_value, url, path, downloaded, int(time.time()), error))
                
                conn.commit()
            finally:
                conn.close()
    
    def get_status(self, hash_value: str) -> Optional[Dict]:
        """获取指定hash的下载状态"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # 查询所有年份的表
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'images_%'
                """)
                
                for (table_name,) in cursor.fetchall():
                    cursor.execute(f"""
                        SELECT url, path, downloaded, timestamp, error
                        FROM {table_name}
                        WHERE hash = ?
                    """, (hash_value,))
                    
                    row = cursor.fetchone()
                    if row:
                        return {
                            'url': row[0],
                            'path': row[1],
                            'downloaded': bool(row[2]),
                            'timestamp': row[3],
                            'error': row[4]
                        }
                return None
            finally:
                conn.close()
    
    def get_stats(self) -> Dict:
        """获取下载统计信息"""
        print("\n=== 数据库统计信息 ===")
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                stats = {
                    'covers': {'total': 0, 'downloaded': 0, 'failed': 0},
                    'avatars': {'total': 0, 'downloaded': 0, 'failed': 0}
                }
                
                # 获取所有年份的表
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'images_%'
                """)
                
                print("查询所有图片相关的表...")
                for (table_name,) in cursor.fetchall():
                    print(f"\n处理表: {table_name}")
                    # 从表名解析类型
                    if 'covers' in table_name:
                        type_name = 'covers'
                    elif 'avatars' in table_name:
                        type_name = 'avatars'
                    else:
                        print(f"跳过不相关的表: {table_name}")
                        continue
                    
                    # 获取统计信息
                    cursor.execute(f"""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN downloaded = 1 THEN 1 ELSE 0 END) as downloaded,
                            SUM(CASE WHEN downloaded = 0 AND error IS NOT NULL THEN 1 ELSE 0 END) as failed
                        FROM {table_name}
                    """)
                    
                    row = cursor.fetchone()
                    if row:
                        print(f"{type_name} 统计结果:")
                        print(f"- 总数: {row[0] or 0}")
                        print(f"- 已下载: {row[1] or 0}")
                        print(f"- 失败: {row[2] or 0}")
                        
                        stats[type_name]['total'] += row[0] or 0
                        stats[type_name]['downloaded'] += row[1] or 0
                        stats[type_name]['failed'] += row[2] or 0
                    
                    # 获取最近下载时间
                    cursor.execute(f"""
                        SELECT MAX(timestamp)
                        FROM {table_name}
                        WHERE downloaded = 1
                    """)
                    
                    last_download = cursor.fetchone()[0]
                    if last_download:
                        current_last = stats[type_name].get('last_download', 0)
                        stats[type_name]['last_download'] = max(current_last, last_download)
                        print(f"最近下载时间: {datetime.fromtimestamp(stats[type_name]['last_download'])}")

                
                return stats
            finally:
                conn.close()
    
    def clear_all(self):
        """清空所有下载状态"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # 获取所有图片相关的表
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'images_%'
                """)
                
                # 删除所有表
                for (table_name,) in cursor.fetchall():
                    cursor.execute(f"DROP TABLE {table_name}")
                
                conn.commit()
            finally:
                conn.close()
    
    def get_failed_downloads(self, type: str, year: int) -> List[Dict]:
        """获取失败的下载记录
        
        Args:
            type: 图片类型 (cover 或 avatar)
            year: 年份
            
        Returns:
            List[Dict]: 失败记录列表
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                table_name = self._get_table_name(type, year)
                
                cursor.execute(f"""
                    SELECT hash, url, path, timestamp, error
                    FROM {table_name}
                    WHERE downloaded = 0 AND error IS NOT NULL
                """)
                
                return [{
                    'hash': row[0],
                    'url': row[1],
                    'path': row[2],
                    'timestamp': row[3],
                    'error': row[4]
                } for row in cursor.fetchall()]
            finally:
                conn.close()

class ImageDownloader:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ImageDownloader, cls).__new__(cls)
                # 将初始化移到这里，确保只执行一次
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化实例属性"""
        if hasattr(self, '_initialized'):
            return
            
        print("\n=== 初始化图片下载器 ===")
        self.session = self._create_session()
        self.download_queue = Queue()
        self.lock = threading.Lock()
        self.is_downloading = False
        
        # 初始化数据库
        self.db = DownloadStatusDB()
        
        # 总下载数量跟踪
        self.total_covers_to_download = 0
        self.total_avatars_to_download = 0
        
        # 添加初始total值的存储
        self.initial_covers_total = 0
        self.initial_avatars_total = 0
        
        # 确保目录存在
        self.base_path = get_output_path('images')
        self.covers_path = os.path.join(self.base_path, 'covers')
        self.avatars_path = os.path.join(self.base_path, 'avatars')
        self.orphaned_covers_path = os.path.join(self.base_path, 'orphaned_covers')
        self.orphaned_avatars_path = os.path.join(self.base_path, 'orphaned_avatars')
        
        os.makedirs(self.covers_path, exist_ok=True)
        os.makedirs(self.avatars_path, exist_ok=True)
        os.makedirs(self.orphaned_covers_path, exist_ok=True)
        os.makedirs(self.orphaned_avatars_path, exist_ok=True)
        
        print("初始化图片下载器完成")
        print(f"图片保存基础路径: {self.base_path}")
        print(f"封面保存路径: {self.covers_path}")
        print(f"头像保存路径: {self.avatars_path}")
        print(f"孤立封面保存路径: {self.orphaned_covers_path}")
        print(f"孤立头像保存路径: {self.orphaned_avatars_path}")
        
        self._initialized = True
    
    def __init__(self):
        """空的初始化方法，实际初始化在_initialize中完成"""
        pass
    
    def _create_session(self) -> requests.Session:
        """创建请求会话，配置重试策略和请求头"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置请求头
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com'
        })
        
        print("创建请求会话完成，已配置重试策略和请求头")
        return session
    
    def _get_file_hash(self, url: str) -> str:
        """获取URL的哈希值作为文件名"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_file_extension(self, url: str) -> str:
        """从URL中获取文件扩展名"""
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        
        # 如果URL中没有扩展名，从Content-Type中获取
        if not ext:
            try:
                response = self.session.head(url, timeout=5)
                content_type = response.headers.get('content-type', '')
                if 'gif' in content_type:
                    return '.gif'
                elif 'png' in content_type:
                    return '.png'
                elif 'jpeg' in content_type or 'jpg' in content_type:
                    return '.jpg'
                elif 'webp' in content_type:
                    return '.webp'
            except Exception as e:
                print(f"获取Content-Type失败: {str(e)}")
        
        return ext if ext else '.jpg'
    
    def _download_image(self, url: str, save_path: str) -> bool:
        """下载单个图片"""
        try:
            print(f"\n开始下载图片: {url}")
            print(f"保存路径: {save_path}")
            
            # 随机延迟
            delay = random.uniform(2, 5)
            print(f"随机延迟 {delay:.2f} 秒")
            time.sleep(delay)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # 检查是否是图片
            content_type = response.headers.get('content-type', '')
            print(f"响应Content-Type: {content_type}")
            
            if not content_type.startswith('image/'):
                print(f"警告：URL {url} 返回的不是图片（{content_type}）")
                return False
            
            # 特殊处理GIF图片
            if 'gif' in content_type.lower():
                print("检测到GIF图片，使用二进制模式保存")
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                if not save_path.lower().endswith('.gif'):
                    new_path = os.path.splitext(save_path)[0] + '.gif'
                    os.rename(save_path, new_path)
                    print(f"重命名文件为GIF格式: {new_path}")
                    return True
            
            # 保存其他类型的图片
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(save_path)
            print(f"图片下载成功，文件大小: {file_size/1024:.2f}KB")
            return True
            
        except Exception as e:
            print(f"下载图片失败 {url}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException):
                print(f"请求异常详情: {e.response.status_code if e.response else '无响应'}")
            return False
    
    def _get_cover_path(self, url: str, year: int = None) -> str:
        """获取封面图片的保存路径，使用哈希值组织目录结构"""
        file_hash = self._get_file_hash(url)
        ext = self._get_file_extension(url)
        sub_dir = file_hash[:2]  # 使用哈希的前两位作为子目录
        save_dir = os.path.join(self.covers_path, str(year), sub_dir) if year else os.path.join(self.covers_path, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{file_hash}{ext}")
    
    def _get_avatar_path(self, url: str, year: int = None) -> str:
        """获取头像图片的保存路径"""
        file_hash = self._get_file_hash(url)
        ext = self._get_file_extension(url)
        sub_dir = file_hash[:2]  # 使用哈希的前两位作为子目录
        save_dir = os.path.join(self.avatars_path, str(year), sub_dir) if year else os.path.join(self.avatars_path, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{file_hash}{ext}")
    
    def download_worker(self):
        """下载工作线程"""
        thread_name = threading.current_thread().name
        print(f"\n=== 线程 {thread_name} 开始工作 ===")
        
        while True:
            try:
                # 使用超时获取任务,避免永久阻塞
                try:
                    task = self.download_queue.get(timeout=1)
                except Empty:
                    # 如果队列为空且下载已完成,退出线程
                    if not self.is_downloading:
                        print(f"线程 {thread_name} 检测到下载完成标志，退出")
                        break
                    continue
                
                if task is None:
                    print(f"线程 {thread_name} 收到停止信号，退出")
                    break
                
                url, save_path, is_cover, hash_value, year = task
                print(f"\n=== 线程 {thread_name} 处理下载任务 ===")
                print(f"URL: {url}")
                print(f"保存路径: {save_path}")
                print(f"类型: {'封面' if is_cover else '头像'}")
                print(f"年份: {year}")
                
                try:
                    # 检查是否已下载
                    status = self.db.get_status(hash_value)
                    if status and status['downloaded']:
                            print(f"跳过已下载的图片: {url}")
                            self.download_queue.task_done()
                            continue
                    
                    # 确保目录存在
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 下载图片
                    print("开始下载图片...")
                    success = self._download_image(url, save_path)
                    print(f"下载结果: {'成功' if success else '失败'}")
                    
                    # 更新状态
                    print("更新下载状态...")
                    self.db.update_status(
                        hash_value=hash_value,
                        type='cover' if is_cover else 'avatar',
                        url=url,
                        path=save_path,
                        downloaded=success,
                        error=None if success else "Download failed",
                        year=year
                    )
                
                except Exception as e:
                    print(f"\n处理下载任务时出错:")
                    print(f"URL: {url}")
                    print(f"错误类型: {type(e).__name__}")
                    print(f"错误信息: {str(e)}")
                    print("错误堆栈:")
                    import traceback
                    print(traceback.format_exc())
                    
                    # 记录错误状态
                    self.db.update_status(
                        hash_value=hash_value,
                        type='cover' if is_cover else 'avatar',
                        url=url,
                        path=save_path,
                        downloaded=False,
                        error=str(e),
                        year=year
                    )
                
                finally:
                    # 无论成功与否，都标记任务完成
                    self.download_queue.task_done()
                    print("任务已标记为完成")
                
            except Exception as e:
                print(f"\n下载工作线程发生错误:")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
                print("错误堆栈:")
                import traceback
                print(traceback.format_exc())
    
    def start_download(self, year: Optional[int] = None):
        """开始下载指定年份的图片"""
        try:
            print(f"\n{'='*50}")
            print(f"开始下载图片 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"目标年份: {'所有年份' if year is None else year}")
            
            # 重置总下载数量
            with self.lock:
                self.total_covers_to_download = 0
                self.total_avatars_to_download = 0
            
            # 设置下载状态标志
            self.is_downloading = True
            
            # 获取要处理的年份列表
            if year:
                years = [year]
            else:
                years = get_available_years()
            
            if not years:
                print("错误：未找到任何可用年份")
                raise ValueError("未找到任何可用年份")
            
            print(f"将处理以下年份: {years}")
            
            # 获取当前已下载的数量
            current_stats = self.db.get_stats()
            current_covers_downloaded = current_stats['covers']['downloaded']
            current_avatars_downloaded = current_stats['avatars']['downloaded']
            
            # 预先计算总下载数量
            total_cover_urls = []
            total_avatar_urls = []
            for year in years:
                cover_urls, avatar_urls = self._preprocess_year_data(year)
                total_cover_urls.extend(cover_urls)
                total_avatar_urls.extend(avatar_urls)
            
            # 过滤出需要下载的新URL
            new_cover_urls = self._filter_new_urls(total_cover_urls, True)
            new_avatar_urls = self._filter_new_urls(total_avatar_urls, False)
            
            # 更新总下载数量
            with self.lock:
                self.total_covers_to_download = len(new_cover_urls)
                self.total_avatars_to_download = len(new_avatar_urls)
                
                # 保存初始total值 = 当前已下载 + 计划下载
                self.initial_covers_total = current_covers_downloaded + self.total_covers_to_download
                self.initial_avatars_total = current_avatars_downloaded + self.total_avatars_to_download
                print(f"\n保存初始total值:")
                print(f"- 封面: {self.initial_covers_total} (已下载: {current_covers_downloaded} + 计划: {self.total_covers_to_download})")
                print(f"- 头像: {self.initial_avatars_total} (已下载: {current_avatars_downloaded} + 计划: {self.total_avatars_to_download})")
            
            print(f"\n总计需要下载的图片:")
            print(f"封面: {self.total_covers_to_download}/{len(total_cover_urls)} 个")
            print(f"头像: {self.total_avatars_to_download}/{len(total_avatar_urls)} 个")
            
            # 如果没有需要下载的图片，直接返回
            if self.total_covers_to_download == 0 and self.total_avatars_to_download == 0:
                print("\n没有新的图片需要下载，任务完成")
                self.is_downloading = False
                return
            
            # 创建下载线程池
            num_threads = 20  # 增加到20个线程
            print(f"\n=== 创建下载线程池 ===")
            print(f"启动 {num_threads} 个下载线程")
            print(f"每个线程随机延迟: 2-5秒")
            print(f"预计最大并发下载数: {num_threads}")
            print("========================\n")
            
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=self.download_worker, name=f"DownloadWorker-{i+1}")
                t.daemon = True  # 设置为守护线程
                t.start()
                threads.append(t)
                print(f"线程 DownloadWorker-{i+1} 已启动")
            
            # 处理每个年份的数据
            total_processed = 0
            for year in years:
                print(f"\n=== 处理 {year} 年数据 ===")
                # 预处理年份数据
                cover_urls, avatar_urls = self._preprocess_year_data(year)
                
                # 过滤出需要下载的新URL
                year_new_cover_urls = self._filter_new_urls(cover_urls, True)
                year_new_avatar_urls = self._filter_new_urls(avatar_urls, False)
                
                print(f"\n{year}年需要下载的图片:")
                print(f"封面: {len(year_new_cover_urls)}/{len(cover_urls)} 个")
                print(f"头像: {len(year_new_avatar_urls)}/{len(avatar_urls)} 个")
                
                # 添加下载任务
                for url in year_new_cover_urls:
                    save_path = self._get_cover_path(url, year)
                    hash_value = self._get_file_hash(url)
                    self.download_queue.put((
                        url,
                        save_path,
                        True,
                        hash_value,
                        year
                    ))
                
                for url in year_new_avatar_urls:
                    save_path = self._get_avatar_path(url, year)
                    hash_value = self._get_file_hash(url)
                    self.download_queue.put((
                        url,
                        save_path,
                        False,
                        hash_value,
                        year
                    ))
                
                total_processed += len(year_new_cover_urls) + len(year_new_avatar_urls)
                
                # 等待队列处理完毕,但设置超时避免永久阻塞
                print("\n等待当前年份下载任务完成...")
                while not self.download_queue.empty():
                    try:
                        self.download_queue.join()
                    except Exception as e:
                        print(f"等待任务完成时出错: {str(e)}")
                        break
            
            print("\n所有数据处理完成，正在停止下载线程...")
            
            # 停止工作线程
            self.is_downloading = False
            for _ in threads:
                self.download_queue.put(None)
            
            # 等待线程结束,但设置超时
            for t in threads:
                t.join(timeout=5)
            
            # 确保状态标记为已完成
            self.is_downloading = False
            
            print("\n=== 下载统计 ===")
            stats = self.get_download_stats()
            print(f"封面图片:")
            print(f"- 总数: {stats['covers']['total']}")
            print(f"- 已下载: {stats['covers']['downloaded']}")
            print(f"- 失败: {stats['covers']['failed']}")
            print(f"\n头像图片:")
            print(f"- 总数: {stats['avatars']['total']}")
            print(f"- 已下载: {stats['avatars']['downloaded']}")
            print(f"- 失败: {stats['avatars']['failed']}")
            
            print(f"\n下载任务完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总处理记录数: {total_processed}")
            print('='*50)
            
        except Exception as e:
            print("\n下载过程发生错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            import traceback
            print("错误堆栈:")
            print(traceback.format_exc())
        finally:
            # 确保无论发生什么情况，状态都会设置为已完成
            print("\n结束下载任务，设置状态为已完成")
            self.is_downloading = False
            
            # 如果下载已完成且所有图片都已下载，重置计数器
            if self.total_covers_to_download == 0 and self.total_avatars_to_download == 0:
                print("下载已完成，重置计数器")
                self.initial_covers_total = 0
                self.initial_avatars_total = 0
            
            if 'conn' in locals():
                conn.close()
                print("数据库连接已关闭")
    
    def get_download_stats(self) -> Dict:
        """获取下载统计信息"""
        print("\n=== 获取下载统计 ===")
        
        # 从数据库获取统计信息
        print("从数据库获取统计信息...")
        stats = self.db.get_stats()
        
        # 如果没有正在下载，计算潜在的下载数量
        if not self.is_downloading:
            print("\n计算潜在的下载数量...")
            # 获取所有年份
            years = get_available_years()
            if years:
                # 收集所有URL
                total_cover_urls = []
                total_avatar_urls = []
                for year in years:
                    cover_urls, avatar_urls = self._preprocess_year_data(year)
                    total_cover_urls.extend(cover_urls)
                    total_avatar_urls.extend(avatar_urls)
                
                # 过滤出需要下载的新URL
                new_cover_urls = self._filter_new_urls(total_cover_urls, True)
                new_avatar_urls = self._filter_new_urls(total_avatar_urls, False)
                
                # 更新总下载数量
                self.total_covers_to_download = len(new_cover_urls)
                self.total_avatars_to_download = len(new_avatar_urls)
                
                # 更新初始total值 - 仅在非下载状态下
                self.initial_covers_total = stats['covers']['downloaded'] + self.total_covers_to_download
                self.initial_avatars_total = stats['avatars']['downloaded'] + self.total_avatars_to_download
                
                print(f"潜在下载数量:")
                print(f"- 封面: {self.total_covers_to_download}")
                print(f"- 头像: {self.total_avatars_to_download}")
                
                print(f"更新初始total值:")
                print(f"- 封面: {self.initial_covers_total}")
                print(f"- 头像: {self.initial_avatars_total}")
        else:
            # 如果正在下载，根据初始total和当前下载量计算待下载数量
            # 初始total应该保持不变，只调整total_planned
            self.total_covers_to_download = max(0, self.initial_covers_total - stats['covers']['downloaded'])
            self.total_avatars_to_download = max(0, self.initial_avatars_total - stats['avatars']['downloaded'])
            
            print(f"下载进度:")
            print(f"- 封面: 已下载 {stats['covers']['downloaded']}/{self.initial_covers_total}，剩余 {self.total_covers_to_download}")
            print(f"- 头像: 已下载 {stats['avatars']['downloaded']}/{self.initial_avatars_total}，剩余 {self.total_avatars_to_download}")
            
            # 检查下载是否实际已完成
            if (self.total_covers_to_download == 0 and self.total_avatars_to_download == 0 and
                stats['covers']['downloaded'] == self.initial_covers_total and
                stats['avatars']['downloaded'] == self.initial_avatars_total):
                print("\n检测到所有图片已下载完成，自动设置下载状态为已完成")
                self.is_downloading = False
        
        # 更新统计信息
        # 对于下载状态，使用初始固定total值
        # 对于非下载状态，使用downloaded + total_planned
        if self.is_downloading:
            stats['covers']['total'] = self.initial_covers_total
            stats['avatars']['total'] = self.initial_avatars_total
        else:
            stats['covers']['total'] = stats['covers']['downloaded'] + self.total_covers_to_download
            stats['avatars']['total'] = stats['avatars']['downloaded'] + self.total_avatars_to_download
        
        print(f"最终统计结果:")
        print(f"- 封面总数: {stats['covers']['total']}")
        print(f"- 封面已下载: {stats['covers']['downloaded']}")
        print(f"- 封面待下载: {self.total_covers_to_download}")
        print(f"- 头像总数: {stats['avatars']['total']}")
        print(f"- 头像已下载: {stats['avatars']['downloaded']}")
        print(f"- 头像待下载: {self.total_avatars_to_download}")
        print(f"- 下载状态: {'正在下载' if self.is_downloading else '已完成'}")
        
        # 添加总下载数量信息
        stats['covers']['total_planned'] = self.total_covers_to_download
        stats['avatars']['total_planned'] = self.total_avatars_to_download
        
        # 添加其他信息
        stats.update({
            'last_update': int(time.time()),
            'is_downloading': self.is_downloading
        })

        
        return stats

    def clear_all_images(self):
        """清空所有图片和下载状态"""
        try:
            print("\n开始清空所有图片和下载状态...")
            
            # 停止当前下载任务（如果有）
            self.is_downloading = False
            self.download_queue = Queue()
            
            # 清空数据库
            self.db.clear_all()
            
            # 删除所有图片文件夹中的文件
            def remove_files_in_dir(directory):
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory, topdown=False):
                        for name in files:
                            try:
                                file_path = os.path.join(root, name)
                                os.remove(file_path)
                            except Exception as e:
                                print(f"删除文件失败: {file_path} - {str(e)}")
                        for name in dirs:
                            try:
                                dir_path = os.path.join(root, name)
                                os.rmdir(dir_path)
                            except Exception as e:
                                print(f"删除文件夹失败: {dir_path} - {str(e)}")
            
            # 清空各个图片目录
            remove_files_in_dir(self.covers_path)
            remove_files_in_dir(self.avatars_path)
            remove_files_in_dir(self.orphaned_covers_path)
            remove_files_in_dir(self.orphaned_avatars_path)
            
            # 重新创建基础目录
            os.makedirs(self.covers_path, exist_ok=True)
            os.makedirs(self.avatars_path, exist_ok=True)
            os.makedirs(self.orphaned_covers_path, exist_ok=True)
            os.makedirs(self.orphaned_avatars_path, exist_ok=True)
            
            print("\n清空操作完成！")
            print(f"- 图片保存基础路径: {self.base_path}")
            print(f"- 封面保存路径: {self.covers_path}")
            print(f"- 头像保存路径: {self.avatars_path}")
            print(f"- 孤立封面保存路径: {self.orphaned_covers_path}")
            print(f"- 孤立头像保存路径: {self.orphaned_avatars_path}")
            
            return True
            
        except Exception as e:
            print(f"\n清空过程发生错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            import traceback
            print("错误堆栈:")
            print(traceback.format_exc())
            return False

    def _preprocess_year_data(self, year: int) -> Tuple[List[str], List[str]]:
        """预处理指定年份的数据，获取需要下载的图片URL
        
        Args:
            year: 要处理的年份
            
        Returns:
            Tuple[List[str], List[str]]: 封面URL列表和头像URL列表
        """
        print(f"\n=== 预处理 {year} 年数据 ===")
        
        conn = get_db()
        try:
            cursor = conn.cursor()
            table_name = f"bilibili_history_{year}"
            
            # 获取封面URL和头像URL，明确包含covers字段
            cursor.execute(f"""
                SELECT DISTINCT cover, covers, author_face
                FROM {table_name}
                WHERE cover IS NOT NULL OR covers IS NOT NULL OR author_face IS NOT NULL
            """)
            
            cover_urls = set()
            avatar_urls = set()
            
            for row in cursor.fetchall():
                cover, covers_json, avatar = row
                
                # 处理封面URL
                if cover:
                    cover_urls.add(cover)
                
                # 处理JSON格式的covers字段
                if covers_json:
                    try:
                        # 先检查covers_json是否已经是JSON对象
                        if isinstance(covers_json, str):
                            covers = json.loads(covers_json)
                        else:
                            covers = covers_json
                            
                        # 如果covers是列表，添加所有URL
                        if isinstance(covers, list):
                            for url in covers:
                                if url and isinstance(url, str):
                                    cover_urls.add(url)
                        # 如果covers是字典，提取其中的所有URL
                        elif isinstance(covers, dict):
                            for key, value in covers.items():
                                if isinstance(value, str) and ('http://' in value or 'https://' in value):
                                    cover_urls.add(value)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"解析covers字段失败: {e} - 值: {covers_json}")
                
                # 处理头像URL
                if avatar:
                    avatar_urls.add(avatar)
            
            print(f"找到 {len(cover_urls)} 个封面URL")
            print(f"找到 {len(avatar_urls)} 个头像URL")
            
            return list(cover_urls), list(avatar_urls)
            
        except sqlite3.Error as e:
            print(f"数据库错误: {str(e)}")
            return [], []
        finally:
            conn.close()

    def _filter_new_urls(self, urls: List[str], is_cover: bool) -> List[str]:
        """过滤出需要下载的新URL
        
        Args:
            urls: URL列表
            is_cover: 是否是封面图片
            
        Returns:
            List[str]: 需要下载的URL列表
        """
        new_urls = []
        for url in urls:
            if not url:  # 跳过空URL
                continue
                
            # 计算URL的哈希值
            hash_value = self._get_file_hash(url)
            
            # 检查是否已下载
            status = self.db.get_status(hash_value)
            if not status or not status['downloaded']:
                new_urls.append(url)
                
        return new_urls

    def stop_download(self):
        """停止当前下载任务"""
        print("\n=== 正在停止下载任务 ===")
        
        # 设置停止标志
        self.is_downloading = False
        
        # 重置下载相关的初始值
        self.total_covers_to_download = 0
        self.total_avatars_to_download = 0
        self.initial_covers_total = 0
        self.initial_avatars_total = 0
        
        # 清空下载队列
        while not self.download_queue.empty():
            try:
                self.download_queue.get_nowait()
                self.download_queue.task_done()
            except Empty:
                break
        
        print("已清空下载队列")
        print("等待现有下载任务完成...")
        print("注意：已开始的下载会继续完成")
        print("======================\n")
        
        return {
            "status": "success",
            "message": "下载任务已停止",
            "stats": self.get_download_stats()
        }

def get_db():
    """获取数据库连接"""
    db_path = get_output_path(config['db_file'])
    return sqlite3.connect(db_path)

def get_available_years() -> List[int]:
    """获取可用的年份列表"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'bilibili_history_%'
        """)
        tables = cursor.fetchall()
        years = []
        for (table_name,) in tables:
            try:
                year = int(table_name.split('_')[-1])
                years.append(year)
            except (ValueError, IndexError):
                continue
        return sorted(years, reverse=True)
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    try:
        print("\n=== 开始图片下载任务 ===")
        print("初始化下载器...")
        downloader = ImageDownloader()
        
        print("\n下载2025年的图片...")
        downloader.start_download(2025)
        
        print("\n获取下载统计...")
        stats = downloader.get_download_stats()
        print("\n下载统计:")
        print(f"封面图片: 总数 {stats['covers']['total']}, "
              f"已下载 {stats['covers']['downloaded']}, "
              f"失败 {stats['covers']['failed']}")
        print(f"头像图片: 总数 {stats['avatars']['total']}, "
              f"已下载 {stats['avatars']['downloaded']}, "
              f"失败 {stats['avatars']['failed']}") 
    except Exception as e:
        print("\n程序执行出错:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print("错误堆栈:")
        print(traceback.format_exc()) 