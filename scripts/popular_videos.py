#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bilibili_popular_videos.py - 获取B站热门视频列表

该脚本用于获取哔哩哔哩当前热门视频列表，使用了WBI签名验证
可以作为API接口导入到其他模块中使用
"""

import json
import os
import time
import random
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests

from scripts.wbi_sign import get_wbi_sign
from scripts.utils import get_output_path, get_database_path

# API 地址
POPULAR_API = "https://api.bilibili.com/x/web-interface/popular"

def get_db_connection(year=None):
    """
    获取数据库连接，支持按年切分数据库

    Args:
        year: 指定年份，如果为None则使用当前年份

    Returns:
        SQLite数据库连接
    """
    # 如果未指定年份，使用当前年份
    if year is None:
        year = datetime.now().year

    # 构建基于年份的数据库路径
    db_filename = f"bilibili_popular_{year}.db"
    db_path = get_database_path(db_filename)
    conn = sqlite3.connect(db_path)

    # 创建表
    create_tables(conn)

    print(f"已连接到{year}年的数据库: {db_path}")
    return conn

# 获取当前年份的数据库连接
def get_current_db_connection():
    """获取当前年份的数据库连接"""
    current_year = datetime.now().year
    return get_db_connection(current_year)

# 获取所有年份的数据库列表
def get_all_year_dbs():
    """获取所有年份的数据库列表"""
    db_dir = os.path.dirname(get_database_path(""))
    db_files = [f for f in os.listdir(db_dir) if f.startswith("bilibili_popular_") and f.endswith(".db")]
    years = []

    for db_file in db_files:
        try:
            year = int(db_file.replace("bilibili_popular_", "").replace(".db", ""))
            years.append(year)
        except ValueError:
            continue

    return sorted(years)

# 获取多年数据库连接
def get_multi_year_connections(start_year=None, end_year=None):
    """获取多年数据库连接"""
    if start_year is None or end_year is None:
        years = get_all_year_dbs()
        if not years:
            # 如果没有找到数据库，返回当前年份
            return {datetime.now().year: get_current_db_connection()}
    else:
        years = range(start_year, end_year + 1)

    connections = {}
    for year in years:
        try:
            connections[year] = get_db_connection(year)
        except Exception as e:
            print(f"连接{year}年数据库出错: {e}")

    return connections

def create_tables(conn):
    """创建数据库表"""
    cursor = conn.cursor()

    # 创建热门视频表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS popular_videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aid TEXT,
        bvid TEXT,
        title TEXT,
        pubdate INTEGER,
        ctime INTEGER,
        desc TEXT,
        videos INTEGER,
        tid INTEGER,
        tname TEXT,
        copyright INTEGER,
        pic TEXT,
        duration INTEGER,
        owner_mid INTEGER,
        owner_name TEXT,
        owner_face TEXT,
        view_count INTEGER,
        danmaku_count INTEGER,
        reply_count INTEGER,
        favorite_count INTEGER,
        coin_count INTEGER,
        share_count INTEGER,
        like_count INTEGER,
        dynamic TEXT,
        cid TEXT,

        /* 展开dimension字段 */
        dimension_width INTEGER,
        dimension_height INTEGER,
        dimension_rotate INTEGER,

        short_link TEXT,
        first_frame TEXT,
        pub_location TEXT,
        cover43 TEXT,
        tidv2 INTEGER,
        tnamev2 TEXT,
        pid_v2 INTEGER,
        pid_name_v2 TEXT,
        season_type INTEGER,
        is_ogv INTEGER,

        /* 展开rights字段 */
        rights_bp INTEGER,
        rights_elec INTEGER,
        rights_download INTEGER,
        rights_movie INTEGER,
        rights_pay INTEGER,
        rights_hd5 INTEGER,
        rights_no_reprint INTEGER,
        rights_autoplay INTEGER,
        rights_ugc_pay INTEGER,
        rights_is_cooperation INTEGER,
        rights_ugc_pay_preview INTEGER,
        rights_no_background INTEGER,
        rights_arc_pay INTEGER,
        rights_pay_free_watch INTEGER,

        /* 展开stat字段 */
        stat_view INTEGER,
        stat_danmaku INTEGER,
        stat_reply INTEGER,
        stat_favorite INTEGER,
        stat_coin INTEGER,
        stat_share INTEGER,
        stat_now_rank INTEGER,
        stat_his_rank INTEGER,
        stat_like INTEGER,
        stat_dislike INTEGER,
        stat_vt INTEGER,
        stat_vv INTEGER,
        stat_fav_g INTEGER,
        stat_like_g INTEGER,

        /* 展开rcmd_reason字段 */
        rcmd_reason_content TEXT,
        rcmd_reason_corner_mark INTEGER,

        ogv_info TEXT,
        enable_vt INTEGER,
        ai_rcmd TEXT,
        fetch_time INTEGER,
        UNIQUE(aid, bvid, fetch_time)
    )
    ''')

    # 创建抓取记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fetch_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fetch_time INTEGER,
        total_fetched INTEGER,
        pages_fetched INTEGER,
        success INTEGER,
        failed_to_save INTEGER DEFAULT 0,
        duplicates_skipped INTEGER DEFAULT 0
    )
    ''')

    # 创建视频热门跟踪表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS popular_video_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aid TEXT,
        bvid TEXT,
        title TEXT,
        first_seen INTEGER,  -- 首次出现时间
        last_seen INTEGER,   -- 最后一次出现时间
        is_active INTEGER,   -- 是否仍在热门列表
        total_duration INTEGER, -- 累计在热门列表的时间(秒)
        highest_rank INTEGER, -- 历史最高排名
        lowest_rank INTEGER,  -- 历史最低排名
        appearances INTEGER DEFAULT 1,  -- 出现次数
        UNIQUE(aid, bvid)
    )
    ''')

    conn.commit()

def insert_video_to_db(conn, video: Dict[str, Any], fetch_time: int, rank: int = 0, auto_commit: bool = False):
    """
    将视频信息插入数据库

    Args:
        conn: 数据库连接
        video: 视频数据
        fetch_time: 抓取时间戳
        rank: 视频排名
        auto_commit: 是否自动提交事务，默认为False，用于批量插入时由外部控制事务
    """
    cursor = conn.cursor()

    # 提取嵌套数据
    owner = video.get('owner', {})
    stat = video.get('stat', {})
    dimension = video.get('dimension', {})
    rcmd_reason = video.get('rcmd_reason', {})
    rights = video.get('rights', {})

    try:
        # 插入到主视频表
        # 计算参数数量
        values = (
            video.get('aid'),
            video.get('bvid'),
            video.get('title'),
            video.get('pubdate'),
            video.get('ctime'),
            video.get('desc'),
            video.get('videos'),
            video.get('tid'),
            video.get('tname'),
            video.get('copyright'),
            video.get('pic'),
            video.get('duration'),
            owner.get('mid'),
            owner.get('name'),
            owner.get('face'),
            stat.get('view'),
            stat.get('danmaku'),
            stat.get('reply'),
            stat.get('favorite'),
            stat.get('coin'),
            stat.get('share'),
            stat.get('like'),
            video.get('dynamic'),
            video.get('cid'),
            # dimension字段展开
            dimension.get('width'),
            dimension.get('height'),
            dimension.get('rotate'),
            video.get('short_link_v2'),  # 使用short_link_v2作为short_link列的值
            video.get('first_frame'),
            video.get('pub_location'),
            video.get('cover43'),
            video.get('tidv2'),
            video.get('tnamev2'),
            video.get('pid_v2'),
            video.get('pid_name_v2'),
            video.get('season_type'),
            1 if video.get('is_ogv') else 0,
            # rights字段展开
            rights.get('bp'),
            rights.get('elec'),
            rights.get('download'),
            rights.get('movie'),
            rights.get('pay'),
            rights.get('hd5'),
            rights.get('no_reprint'),
            rights.get('autoplay'),
            rights.get('ugc_pay'),
            rights.get('is_cooperation'),
            rights.get('ugc_pay_preview'),
            rights.get('no_background'),
            rights.get('arc_pay'),
            rights.get('pay_free_watch'),
            # stat字段展开
            stat.get('view'),
            stat.get('danmaku'),
            stat.get('reply'),
            stat.get('favorite'),
            stat.get('coin'),
            stat.get('share'),
            stat.get('now_rank'),
            stat.get('his_rank'),
            stat.get('like'),
            stat.get('dislike'),
            stat.get('vt'),
            stat.get('vv'),
            stat.get('fav_g'),
            stat.get('like_g'),
            # rcmd_reason字段展开
            rcmd_reason.get('content'),
            rcmd_reason.get('corner_mark'),
            # 其他字段
            json.dumps(video.get('ogv_info', {}), ensure_ascii=False) if video.get('ogv_info') else None,
            video.get('enable_vt'),
            json.dumps(video.get('ai_rcmd', {}), ensure_ascii=False) if video.get('ai_rcmd') else None,
            fetch_time
        )

        cursor.execute('''
        INSERT OR REPLACE INTO popular_videos (
            aid, bvid, title, pubdate, ctime, desc, videos, tid, tname, copyright,
            pic, duration, owner_mid, owner_name, owner_face, view_count, danmaku_count,
            reply_count, favorite_count, coin_count, share_count, like_count, dynamic,
            cid,
            dimension_width, dimension_height, dimension_rotate,
            short_link, first_frame, pub_location, cover43, tidv2,
            tnamev2, pid_v2, pid_name_v2, season_type, is_ogv,
            rights_bp, rights_elec, rights_download, rights_movie, rights_pay,
            rights_hd5, rights_no_reprint, rights_autoplay, rights_ugc_pay,
            rights_is_cooperation, rights_ugc_pay_preview, rights_no_background,
            rights_arc_pay, rights_pay_free_watch,
            stat_view, stat_danmaku, stat_reply, stat_favorite, stat_coin,
            stat_share, stat_now_rank, stat_his_rank, stat_like, stat_dislike,
            stat_vt, stat_vv, stat_fav_g, stat_like_g,
            rcmd_reason_content, rcmd_reason_corner_mark,
            ogv_info, enable_vt, ai_rcmd, fetch_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)

        # 更新跟踪表
        update_tracking_info(conn, video, fetch_time, rank)

        # 只有在auto_commit为True时才提交事务
        if auto_commit:
            conn.commit()

    except sqlite3.Error as e:
        print(f"插入数据库时出错: {e}")
        # 如果auto_commit为True，则回滚操作
        if auto_commit:
            conn.rollback()
        raise

def update_tracking_info(conn, video: Dict[str, Any], fetch_time: int, rank: int = 0, auto_commit: bool = False):
    """
    更新视频热门跟踪信息

    Args:
        conn: 数据库连接
        video: 视频数据
        fetch_time: 抓取时间戳
        rank: 视频排名
        auto_commit: 是否自动提交事务，默认为False，用于批量插入时由外部控制事务
    """
    cursor = conn.cursor()

    aid = video.get('aid')
    bvid = video.get('bvid')
    title = video.get('title')

    try:
        # 检查视频是否已存在于跟踪表中
        cursor.execute(
            "SELECT first_seen, last_seen, is_active, highest_rank, lowest_rank, appearances FROM popular_video_tracking WHERE aid = ? AND bvid = ?",
            (aid, bvid)
        )
        result = cursor.fetchone()

        if result:
            # 视频已存在，更新信息
            first_seen, last_seen, is_active, highest_rank, lowest_rank, appearances = result

            # 更新最后一次出现时间、出现次数和活跃状态
            if last_seen < fetch_time:
                cursor.execute(
                    "UPDATE popular_video_tracking SET last_seen = ?, is_active = 1, appearances = appearances + 1 WHERE aid = ? AND bvid = ?",
                    (fetch_time, aid, bvid)
                )

            # 更新排名记录
            if rank > 0:
                if highest_rank is None or rank < highest_rank:
                    cursor.execute(
                        "UPDATE popular_video_tracking SET highest_rank = ? WHERE aid = ? AND bvid = ?",
                        (rank, aid, bvid)
                    )
                if lowest_rank is None or rank > lowest_rank:
                    cursor.execute(
                        "UPDATE popular_video_tracking SET lowest_rank = ? WHERE aid = ? AND bvid = ?",
                        (rank, aid, bvid)
                    )
        else:
            # 新视频，插入记录
            cursor.execute('''
            INSERT INTO popular_video_tracking (
                aid, bvid, title, first_seen, last_seen, is_active,
                total_duration, highest_rank, lowest_rank, appearances
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                aid, bvid, title, fetch_time, fetch_time, 1,
                0, rank if rank > 0 else None, rank if rank > 0 else None, 1
            ))

        # 只有在auto_commit为True时才提交事务
        if auto_commit:
            conn.commit()

    except sqlite3.Error as e:
        print(f"更新跟踪信息时出错: {e}")
        # 如果auto_commit为True，则回滚操作
        if auto_commit:
            conn.rollback()
        raise

def update_inactive_videos(conn, fetch_time: int):
    """
    更新不再在热门列表中的视频状态

    Args:
        conn: 数据库连接
        fetch_time: 当前抓取时间戳
    """
    cursor = conn.cursor()

    try:
        # 查找所有活跃但在当前抓取中未出现的视频
        cursor.execute('''
        SELECT aid, bvid, last_seen FROM popular_video_tracking
        WHERE is_active = 1 AND last_seen < ?
        ''', (fetch_time,))

        inactive_videos = cursor.fetchall()

        # 更新这些视频的状态
        for aid, bvid, last_seen in inactive_videos:
            total_duration = fetch_time - last_seen

            cursor.execute('''
            UPDATE popular_video_tracking
            SET is_active = 0, total_duration = total_duration + ?
            WHERE aid = ? AND bvid = ?
            ''', (total_duration, aid, bvid))

        return len(inactive_videos)
    except sqlite3.Error as e:
        print(f"更新非活跃视频时出错: {e}")
        return 0

def save_fetch_record(conn, fetch_time: int, total_fetched: int, pages_fetched: int, success: bool, failed: int = 0, duplicates: int = 0):
    """
    保存抓取记录

    Args:
        conn: 数据库连接
        fetch_time: 抓取时间戳
        total_fetched: 总共抓取的视频数
        pages_fetched: 抓取的页数
        success: 是否成功
        failed: 保存失败的数量
        duplicates: 重复跳过的数量
    """
    cursor = conn.cursor()

    try:
        cursor.execute('''
        INSERT INTO fetch_records (fetch_time, total_fetched, pages_fetched, success, failed_to_save, duplicates_skipped)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            fetch_time,
            total_fetched,
            pages_fetched,
            1 if success else 0,
            failed,
            duplicates
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"保存抓取记录时出错: {e}")
        conn.rollback()

def get_popular_videos(page_num: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """
    获取B站当前热门视频列表

    Args:
        page_num: 页码，默认为1
        page_size: 每页视频数量，默认为20

    Returns:
        Dict[str, Any]: 热门视频列表数据
    """
    # 基础参数
    params = {
        "ps": page_size,
        "pn": page_num,
        "web_location": "333.934"  # 网页位置参数
    }

    # 添加 WBI 签名
    signed_params = get_wbi_sign(params)

    # 请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Origin": "https://www.bilibili.com"
    }

    try:
        # 发送请求
        response = requests.get(
            POPULAR_API,
            params=signed_params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        # 解析响应数据
        data = response.json()

        return data
    except requests.RequestException as e:
        print(f"网络请求失败: {e}")
        return {"code": -1, "message": f"网络请求失败: {e}", "ttl": 0, "data": None}
    except json.JSONDecodeError as e:
        print(f"解析JSON数据失败: {e}")
        return {"code": -1, "message": f"解析JSON数据失败: {e}", "ttl": 0, "data": None}
    except Exception as e:
        print(f"获取热门视频列表失败: {e}")
        return {"code": -1, "message": f"获取热门视频列表失败: {e}", "ttl": 0, "data": None}

def get_all_popular_videos(
    page_size: int = 20,
    max_pages: int = 100,
    save_to_db: bool = True,
    progress_callback = None
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    """
    获取B站所有热门视频，直到没有更多数据为止

    Args:
        page_size: 每页视频数量，默认为20
        max_pages: 最大获取页数，防止无限循环，默认为100
        save_to_db: 是否保存到数据库，默认为True
        progress_callback: 进度回调函数，可选。接受当前进度(百分比)、状态消息、当前页码和总页数作为参数

    Returns:
        Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
            - 所有热门视频列表
            - 是否成功获取完所有数据
            - 抓取统计信息
    """
    all_videos = []
    total_items = 0
    page_num = 1
    has_more = True
    conn = None
    fetch_time = int(time.time())
    failed_count = 0
    duplicate_count = 0
    inactive_count = 0
    # 用于收集所有要保存的视频数据
    videos_to_save = []

    try:
        # 如果需要保存到数据库，建立连接
        if save_to_db:
            conn = get_current_db_connection()

        while has_more and page_num <= max_pages:
            print(f"正在获取第 {page_num} 页数据...")

            # 报告进度（如果提供了回调函数）
            estimated_progress = min(95, int((page_num / max_pages) * 100))
            if progress_callback:
                progress_callback(
                    estimated_progress,
                    f"正在获取第 {page_num} 页数据...",
                    page_num,
                    max_pages
                )

            # 随机延迟，模拟人类行为
            delay = random.uniform(3.0, 7.0)
            print(f"等待 {delay:.2f} 秒...")
            time.sleep(delay)

            # 获取当前页数据
            data = get_popular_videos(page_num=page_num, page_size=page_size)

            # 检查是否成功
            if data["code"] != 0 or not data.get("data"):
                print(f"获取第 {page_num} 页数据失败: {data.get('message', '未知错误')}")

                # 报告错误（如果提供了回调函数）
                if progress_callback:
                    progress_callback(
                        estimated_progress,
                        f"获取第 {page_num} 页数据失败: {data.get('message', '未知错误')}",
                        page_num,
                        max_pages,
                        success=False
                    )

                # 保存抓取记录
                if save_to_db and conn:
                    save_fetch_record(conn, fetch_time, len(all_videos), page_num - 1, False, failed_count, duplicate_count)

                fetch_stats = {
                    "status": "error",
                    "message": data.get('message', '未知错误'),
                    "pages_fetched": page_num - 1,
                    "total_items": total_items,
                    "saved_successfully": len(all_videos),
                    "failed_to_save": failed_count,
                    "duplicates_skipped": duplicate_count,
                    "fetch_time": datetime.fromtimestamp(fetch_time).strftime("%Y-%m-%d %H:%M:%S")
                }

                return all_videos, False, fetch_stats

            # 提取视频列表
            video_list = data["data"].get("list", [])
            if video_list:
                total_items += len(video_list)

                # 收集视频数据，但不立即写入数据库
                if save_to_db:
                    for index, video in enumerate(video_list):
                        # 计算当前视频的排名
                        rank = (page_num - 1) * page_size + index + 1
                        # 将视频和排名信息添加到待保存列表
                        videos_to_save.append((video, rank))

                # 提取视频信息并添加到总列表中
                videos = extract_video_info(data)
                all_videos.extend(videos)

                # 输出当前获取进度
                print(f"已获取 {len(all_videos)} 个视频")

                # 报告进度（如果提供了回调函数）
                if progress_callback:
                    progress_callback(
                        estimated_progress,
                        f"已获取 {len(all_videos)} 个视频",
                        page_num,
                        max_pages
                    )

            # 检查是否还有更多数据
            has_more = not data.get("data", {}).get("no_more", True)

            # 准备获取下一页
            if has_more:
                page_num += 1
            else:
                print("已获取全部热门视频数据")

                # 报告进度（如果提供了回调函数）
                if progress_callback:
                    progress_callback(
                        95,
                        "已获取全部热门视频数据，正在处理...",
                        page_num,
                        page_num
                    )

        # 一次性批量保存所有视频数据到数据库
        if save_to_db and conn and videos_to_save:
            print(f"开始一次性保存 {len(videos_to_save)} 个视频到数据库...")

            # 报告进度（如果提供了回调函数）
            if progress_callback:
                progress_callback(
                    96,
                    f"正在保存 {len(videos_to_save)} 个视频到数据库...",
                    page_num,
                    page_num
                )

            # 开始事务
            conn.execute("BEGIN TRANSACTION")

            try:
                cursor = conn.cursor()
                for video, rank in videos_to_save:
                    try:
                        # 检查是否已存在相同视频记录
                        cursor.execute(
                            "SELECT 1 FROM popular_videos WHERE aid = ? AND bvid = ? AND fetch_time = ?",
                            (video.get('aid'), video.get('bvid'), fetch_time)
                        )
                        exists = cursor.fetchone() is not None

                        if exists:
                            duplicate_count += 1
                            print(f"跳过重复视频: {video.get('bvid')} - {video.get('title')}")
                        else:
                            # 插入视频数据，不自动提交事务
                            insert_video_to_db(conn, video, fetch_time, rank, auto_commit=False)
                    except Exception as e:
                        failed_count += 1
                        print(f"保存视频 {video.get('bvid')} 时出错: {e}")

                # 提交事务
                conn.execute("COMMIT")
                print(f"成功保存 {len(videos_to_save) - failed_count - duplicate_count} 个视频到数据库")

            except Exception as e:
                # 回滚事务
                conn.execute("ROLLBACK")
                print(f"批量保存视频时出错，已回滚: {e}")
                failed_count = len(videos_to_save)

            # 报告进度（如果提供了回调函数）
            if progress_callback:
                progress_callback(
                    97,
                    f"已保存 {len(videos_to_save) - failed_count - duplicate_count} 个视频到数据库",
                    page_num,
                    page_num
                )

        # 更新不再活跃的视频
        if save_to_db and conn:
            inactive_count = update_inactive_videos(conn, fetch_time)
            print(f"已更新 {inactive_count} 个不再活跃的视频")

            # 报告进度（如果提供了回调函数）
            if progress_callback:
                progress_callback(
                    98,
                    f"已更新 {inactive_count} 个不再活跃的视频",
                    max_pages,
                    max_pages
                )

        # 保存抓取记录
        if save_to_db and conn:
            save_fetch_record(conn, fetch_time, len(all_videos), page_num - 1, True, failed_count, duplicate_count)

        fetch_stats = {
            "status": "success",
            "total_videos": len(all_videos),
            "fetch_stats": {
                "pages_fetched": page_num - 1,
                "total_items": total_items,
                "saved_successfully": len(all_videos) - failed_count - duplicate_count,
                "failed_to_save": failed_count,
                "duplicates_skipped": duplicate_count,
                "inactive_updated": inactive_count,
                "fetch_time": datetime.fromtimestamp(fetch_time).strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # 报告完成（如果提供了回调函数）
        if progress_callback:
            progress_callback(
                100,
                "热门视频获取完成",
                max_pages,
                max_pages,
                success=True
            )

        return all_videos, True, fetch_stats

    except Exception as e:
        print(f"获取所有热门视频时出错: {e}")

        # 报告错误（如果提供了回调函数）
        if progress_callback:
            progress_callback(
                min(95, int((page_num / max_pages) * 100)),
                f"获取热门视频时出错: {str(e)}",
                page_num,
                max_pages,
                success=False
            )

        # 保存抓取记录
        if save_to_db and conn:
            save_fetch_record(conn, fetch_time, len(all_videos), page_num - 1, False, failed_count, duplicate_count)

        fetch_stats = {
            "status": "error",
            "message": str(e),
            "pages_fetched": page_num - 1,
            "total_items": total_items,
            "saved_successfully": len(all_videos) - failed_count - duplicate_count,
            "failed_to_save": failed_count,
            "duplicates_skipped": duplicate_count,
            "fetch_time": datetime.fromtimestamp(fetch_time).strftime("%Y-%m-%d %H:%M:%S")
        }

        return all_videos, False, fetch_stats
    finally:
        # 关闭数据库连接
        if conn:
            conn.close()

def extract_video_info(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从返回的数据中提取视频基本信息

    Args:
        data: API返回的数据

    Returns:
        List[Dict[str, Any]]: 视频信息列表
    """
    if data["code"] != 0 or not data.get("data") or not data["data"].get("list"):
        return []

    videos = []
    for video in data["data"]["list"]:
        videos.append({
            "aid": video.get("aid"),
            "bvid": video.get("bvid"),
            "title": video.get("title"),
            "author": video.get("owner", {}).get("name"),
            "mid": video.get("owner", {}).get("mid"),
            "play": video.get("stat", {}).get("view"),
            "favorite": video.get("stat", {}).get("favorite"),
            "coin": video.get("stat", {}).get("coin"),
            "share": video.get("stat", {}).get("share"),
            "like": video.get("stat", {}).get("like"),
            "duration": video.get("duration"),
            "pubdate": video.get("pubdate"),
            "description": video.get("desc"),
            "tname": video.get("tname"),
            "short_link": video.get("short_link_v2")
        })

    return videos

def print_popular_videos(video_list: List[Dict[str, Any]], max_display: int = None) -> None:
    """
    打印视频信息

    Args:
        video_list: 视频信息列表
        max_display: 最大显示数量，默认为None，表示显示所有视频
    """
    if not video_list:
        print("未获取到视频信息")
        return

    # 如果指定了最大显示数量，则限制显示
    if max_display is not None and max_display > 0:
        display_list = video_list[:max_display]
        print(f"\n共获取到 {len(video_list)} 个热门视频，显示前 {len(display_list)} 个:")
    else:
        display_list = video_list
        print(f"\n共获取到 {len(video_list)} 个热门视频:")

    print("-" * 80)

    for i, video in enumerate(display_list, 1):
        try:
            # 格式化时间
            pubdate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(video.get("pubdate", 0)))

            # 格式化播放量
            play_count = video.get("play", 0)
            if play_count and play_count > 10000:
                play_count = f"{play_count / 10000:.1f}万"

            print(f"{i}. {video.get('title')}")
            print(f"   UP主: {video.get('author')} (UID: {video.get('mid')})")
            print(f"   播放: {play_count} | 点赞: {video.get('like')} | 投币: {video.get('coin')} | 收藏: {video.get('favorite')}")
            print(f"   BV号: {video.get('bvid')} | 分区: {video.get('tname')} | 发布时间: {pubdate}")
            print(f"   链接: {video.get('short_link')}")
            print("-" * 80)
        except Exception as e:
            print(f"打印视频信息时出错: {e}")
            continue

def query_recent_videos(limit: int = 20) -> List[Dict[str, Any]]:
    """
    从数据库查询最近获取的视频

    Args:
        limit: 限制返回的数量

    Returns:
        List[Dict[str, Any]]: 视频信息列表
    """
    conn = None
    try:
        conn = get_current_db_connection()
        cursor = conn.cursor()

        # 获取最新一次抓取的时间
        cursor.execute('''
        SELECT MAX(fetch_time) FROM fetch_records WHERE success = 1
        ''')
        result = cursor.fetchone()

        if not result or not result[0]:
            return []

        latest_fetch_time = result[0]

        # 查询这次抓取的视频
        cursor.execute('''
        SELECT
            aid, bvid, title, pubdate, owner_mid, owner_name,
            view_count, favorite_count, coin_count, share_count, like_count,
            duration, tname, short_link
        FROM popular_videos
        WHERE fetch_time = ?
        ORDER BY view_count DESC
        LIMIT ?
        ''', (latest_fetch_time, limit))

        rows = cursor.fetchall()

        videos = []
        for row in rows:
            videos.append({
                "aid": row[0],
                "bvid": row[1],
                "title": row[2],
                "pubdate": row[3],
                "mid": row[4],
                "author": row[5],
                "play": row[6],
                "favorite": row[7],
                "coin": row[8],
                "share": row[9],
                "like": row[10],
                "duration": row[11],
                "tname": row[12],
                "short_link": row[13]
            })

        return videos
    except sqlite3.Error as e:
        print(f"查询数据库时出错: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_fetch_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取抓取历史记录

    Args:
        limit: 限制返回的数量

    Returns:
        List[Dict[str, Any]]: 抓取历史列表
    """
    connections = {}
    history = []

    try:
        # 获取所有年份的数据库连接
        connections = get_multi_year_connections()

        # 从每个年份的数据库中查询抓取历史
        for year, conn in connections.items():
            cursor = conn.cursor()
            cursor.execute('''
            SELECT fetch_time, total_fetched, pages_fetched, success
            FROM fetch_records
            ORDER BY fetch_time DESC
            LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            for row in rows:
                fetch_time = row[0]
                history.append({
                    "fetch_time": fetch_time,
                    "fetch_time_str": datetime.fromtimestamp(fetch_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "total_fetched": row[1],
                    "pages_fetched": row[2],
                    "success": bool(row[3]),
                    "year": year
                })

        # 按照fetch_time降序排序
        history.sort(key=lambda x: x["fetch_time"], reverse=True)

        # 限制返回数量
        return history[:limit]
    except sqlite3.Error as e:
        print(f"查询抓取历史时出错: {e}")
        return []
    finally:
        for conn in connections.values():
            if conn:
                conn.close()

def get_video_tracking_stats(limit: int = 20) -> List[Dict[str, Any]]:
    """
    获取视频热门跟踪统计，修复重复视频问题

    Args:
        limit: 限制返回的数量

    Returns:
        List[Dict[str, Any]]: 视频热门统计信息（去重后）
    """
    connections = {}
    stats_dict = {}  # 使用字典存储结果，以bvid为键避免重复

    try:
        # 获取所有年份的数据库连接
        connections = get_multi_year_connections()

        for year, conn in connections.items():
            cursor = conn.cursor()

            # 修改查询，使用DISTINCT消除重复，并确保只选择最新的记录
            cursor.execute('''
            WITH RankedVideos AS (
                SELECT
                    t.aid, t.bvid, t.title, t.first_seen, t.last_seen,
                    t.is_active, t.total_duration, t.highest_rank,
                    t.lowest_rank, t.appearances,
                    p.owner_name,
                    ROW_NUMBER() OVER (PARTITION BY t.bvid ORDER BY t.last_seen DESC) as rn
                FROM popular_video_tracking t
                LEFT JOIN popular_videos p ON t.bvid = p.bvid
                ORDER BY
                    CASE WHEN t.is_active = 1 THEN (? - t.first_seen) + t.total_duration
                         ELSE t.total_duration END DESC
            )
            SELECT * FROM RankedVideos WHERE rn = 1
            LIMIT ?
            ''', (int(time.time()), limit))

            rows = cursor.fetchall()

            for row in rows:
                bvid = row[1]  # 获取视频bvid

                # 如果该视频已经添加过（可能来自其他年份的数据库），则跳过
                if bvid in stats_dict:
                    continue

                first_seen_date = datetime.fromtimestamp(row[3]).strftime("%Y-%m-%d %H:%M:%S")
                last_seen_date = datetime.fromtimestamp(row[4]).strftime("%Y-%m-%d %H:%M:%S")

                # 计算持续时间显示
                total_seconds = row[6]
                if row[5] == 1:  # 如果仍然活跃，加上当前时间差
                    total_seconds += (int(time.time()) - row[4])

                days = total_seconds // (24 * 3600)
                hours = (total_seconds % (24 * 3600)) // 3600
                minutes = (total_seconds % 3600) // 60
                duration_str = ""
                if days > 0:
                    duration_str += f"{days}天"
                if hours > 0 or days > 0:
                    duration_str += f"{hours}小时"
                duration_str += f"{minutes}分钟"

                stats_dict[bvid] = {
                    "aid": row[0],
                    "bvid": bvid,
                    "title": row[2],
                    "first_seen": row[3],
                    "first_seen_str": first_seen_date,
                    "last_seen": row[4],
                    "last_seen_str": last_seen_date,
                    "is_active": bool(row[5]),
                    "total_duration": row[6],
                    "duration_str": duration_str,
                    "highest_rank": row[7],
                    "lowest_rank": row[8],
                    "appearances": row[9],
                    "author": row[10]
                }

        # 将字典值转为列表并排序（按total_duration降序）
        stats = list(stats_dict.values())
        stats.sort(key=lambda x: x["total_duration"], reverse=True)

        # 限制返回数量
        return stats[:limit]

    except sqlite3.Error as e:
        print(f"查询视频跟踪统计时出错: {e}")
        return []
    finally:
        for conn in connections.values():
            if conn:
                conn.close()

def cleanup_inactive_video_records(year: Optional[int] = None):
    """
    清理已经不在热门列表的视频数据，只保留首条和末条记录

    此函数执行以下操作：
    1. 找出所有已经不在热门列表的视频（is_active=0）
    2. 对每个视频，保留其第一条和最后一条记录
    3. 删除该视频的所有中间记录

    Args:
        year: 指定清理的年份；不传则清理所有年份数据库

    Returns:
        dict: 清理统计信息
    """
    connections = {}
    stats = {
        "processed_videos": 0,
        "deleted_records": 0,
        "error_count": 0,
        "year_stats": {}
    }

    try:
        # 获取数据库连接（支持按年清理）
        if year is None:
            connections = get_multi_year_connections()
        else:
            db_filename = f"bilibili_popular_{year}.db"
            db_path = get_database_path(db_filename)
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"未找到{year}年的热门视频数据库文件: {db_path}")
            connections = {year: get_db_connection(year)}

        for db_year, conn in connections.items():
            year_stats = {
                "processed_videos": 0,
                "deleted_records": 0
            }

            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")

            try:
                # 1. 查找所有已经不在热门列表的视频
                cursor.execute("""
                    SELECT bvid
                    FROM popular_video_tracking
                    WHERE is_active = 0
                """)

                inactive_videos = [row[0] for row in cursor.fetchall()]

                print(f"{db_year}年数据库中找到 {len(inactive_videos)} 个不活跃视频")

                for bvid in inactive_videos:
                    # 2. 获取该视频的所有记录时间戳，按时间排序
                    cursor.execute("""
                        SELECT fetch_time
                        FROM popular_videos
                        WHERE bvid = ?
                        ORDER BY fetch_time
                    """, (bvid,))

                    fetch_times = [row[0] for row in cursor.fetchall()]

                    if len(fetch_times) <= 2:
                        # 如果只有两条或更少记录，不需要清理
                        continue

                    # 3. 保留第一条和最后一条记录，删除中间记录
                    first_time = fetch_times[0]
                    last_time = fetch_times[-1]

                    # 构建要删除的时间列表（排除首尾）
                    times_to_delete = fetch_times[1:-1]

                    # 将时间列表转换为IN子句可用的格式
                    placeholders = ','.join(['?'] * len(times_to_delete))

                    # 4. 执行删除
                    cursor.execute(f"""
                        DELETE FROM popular_videos
                        WHERE bvid = ? AND fetch_time IN ({placeholders})
                    """, [bvid] + times_to_delete)

                    deleted_count = cursor.rowcount

                    # 6. 更新统计信息
                    year_stats["processed_videos"] += 1
                    year_stats["deleted_records"] += deleted_count

                    # 7. 输出清理信息
                    if deleted_count > 0:
                        print(f"清理视频 {bvid}: 删除了 {deleted_count} 条记录，保留首条({first_time})和末条({last_time})记录")

                # 提交事务
                cursor.execute("COMMIT")

                # 更新总统计信息
                stats["processed_videos"] += year_stats["processed_videos"]
                stats["deleted_records"] += year_stats["deleted_records"]
                stats["year_stats"][db_year] = year_stats

            except Exception as e:
                # 发生错误，回滚事务
                cursor.execute("ROLLBACK")
                print(f"{db_year}年数据清理时出错: {e}")
                stats["error_count"] += 1
                stats["year_stats"][db_year] = {"error": str(e)}

        print(f"数据清理完成: 处理了 {stats['processed_videos']} 个视频，删除了 {stats['deleted_records']} 条记录")

        # 执行VACUUM操作回收空间
        for db_year, conn in connections.items():
            try:
                print(f"正在对{db_year}年数据库执行VACUUM操作...")
                conn.execute("VACUUM")
                print(f"{db_year}年数据库VACUUM操作完成")
            except Exception as e:
                print(f"{db_year}年数据库VACUUM操作失败: {e}")

        return stats

    except Exception as e:
        print(f"执行数据清理时出错: {e}")
        stats["error"] = str(e)
        return stats
    finally:
        # 关闭所有数据库连接
        for conn in connections.values():
            if conn:
                conn.close()

def schedule_daily_cleanup():
    """
    设置每日数据清理定时任务，应在应用启动时调用此函数
    """
    import threading
    import schedule
    import time

    def run_cleanup():
        print(f"===== 开始执行每日数据清理，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        stats = cleanup_inactive_video_records()
        print(f"===== 每日数据清理完成，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        print(f"清理统计: {stats}")

    def run_scheduler():
        # 设置每天凌晨3点执行清理
        schedule.every().day.at("03:00").do(run_cleanup)

        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

    # 创建并启动调度器线程
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # 设为守护线程，主程序退出时自动结束
    scheduler_thread.start()

    print("已设置每日数据清理定时任务，将在每天凌晨3点执行")

    # 也可以提供一个立即执行的选项
    return scheduler_thread

def main() -> None:
    """主函数"""
    try:
        print("正在获取B站热门视频列表...")

        # 默认获取所有热门视频
        videos, success, fetch_stats = get_all_popular_videos(page_size=20)

        if success and videos:
            # 只显示前20个视频
            print_popular_videos(videos, max_display=20)

            # 提示用户是否要显示所有视频
            if len(videos) > 20:
                choice = input("\n是否显示所有获取到的视频？(y/n): ")
                if choice.lower() == 'y':
                    print_popular_videos(videos)

        elif not videos:
            print("未获取到任何视频数据")

    except KeyboardInterrupt:
        print("\n程序已被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        print("\n程序执行完毕")

if __name__ == "__main__":
    main()
