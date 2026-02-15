import sqlite3
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from scripts.utils import load_config, get_output_path

router = APIRouter()
config = load_config()

def get_db():
    """获取数据库连接"""
    db_path = get_output_path(config['db_file'])
    return sqlite3.connect(db_path)

def validate_year_and_get_table(year: Optional[int]) -> tuple:
    """验证年份并获取对应的表名"""
    from scripts.analyze_bilibili_history import get_available_years
    
    # 获取可用年份列表
    available_years = get_available_years()
    if not available_years:
        return None, None, {
            "status": "error",
            "message": "未找到任何历史记录数据"
        }
    
    # 如果未指定年份，使用最新的年份；如果指定年份不存在，则自动回退到最新可用年份
    if year is None:
        target_year = available_years[0]
    elif year in available_years:
        target_year = year
    else:
        target_year = available_years[0]
    
    table_name = f"bilibili_history_{target_year}"
    return table_name, target_year, available_years

def analyze_popular_hit_rate(cursor, table_name: str, target_year: int) -> dict:
    """分析热门视频命中率"""
    
    # 1. 获取用户观看的所有视频
    cursor.execute(f"""
        SELECT DISTINCT bvid, title, author_name, view_at, duration, progress
        FROM {table_name}
        WHERE bvid IS NOT NULL AND bvid != ''
        ORDER BY view_at ASC
    """)
    
    user_videos = cursor.fetchall()
    total_watched = len(user_videos)
    
    if total_watched == 0:
        return {
            "total_watched": 0,
            "popular_hit_count": 0,
            "hit_rate": 0,
            "insights": ["本年度没有观看记录"]
        }
    
    # 2. 获取热门视频数据库连接
    popular_connections = {}
    try:
        # 获取所有年份的热门视频数据库连接
        from scripts.popular_videos import get_multi_year_connections
        popular_connections = get_multi_year_connections()
    except Exception as e:
        print(f"获取热门视频数据库连接失败: {e}")
        return {
            "total_watched": total_watched,
            "popular_hit_count": 0,
            "hit_rate": 0,
            "insights": [f"观看了 {total_watched} 个视频，但无法获取热门视频数据进行对比"]
        }
    
    # 3. 检查哪些视频曾经是热门，并获取时间信息
    popular_hits = []
    popular_bvids = set()
    popular_video_times = {}  # bvid -> {"pubdate": timestamp}

    for year, conn in popular_connections.items():
        try:
            pop_cursor = conn.cursor()
            # 查询所有热门视频的bvid和发布时间
            pop_cursor.execute("SELECT DISTINCT bvid, pubdate FROM popular_videos WHERE bvid IS NOT NULL AND pubdate IS NOT NULL")
            for row in pop_cursor.fetchall():
                bvid, pubdate = row
                popular_bvids.add(bvid)
                if bvid not in popular_video_times:
                    popular_video_times[bvid] = {"pubdate": pubdate}
        except Exception as e:
            print(f"查询 {year} 年热门视频数据失败: {e}")
            continue

    # 4. 统计命中的热门视频并分析观看时机
    time_patterns = {
        "immediate_watch": 0,  # 发布后立即观看（7天内）
        "trending_watch": 0,   # 热门期观看（7天后）
        "unknown_timing": 0    # 无法确定时机
    }

    import datetime

    for video in user_videos:
        bvid = video[0]
        view_timestamp = video[3]

        if bvid in popular_bvids:
            popular_hits.append({
                "bvid": bvid,
                "title": video[1],
                "author": video[2],
                "view_at": video[3],
                "duration": video[4],
                "progress": video[5]
            })

            # 分析观看时机
            if bvid in popular_video_times and popular_video_times[bvid]["pubdate"]:
                try:
                    view_date = datetime.datetime.fromtimestamp(view_timestamp)
                    pub_date = datetime.datetime.fromtimestamp(popular_video_times[bvid]["pubdate"])
                    days_diff = (view_date - pub_date).days

                    if days_diff <= 7:
                        time_patterns["immediate_watch"] += 1
                    else:
                        time_patterns["trending_watch"] += 1

                except Exception as e:
                    print(f"处理视频 {bvid} 时间数据失败: {e}")
                    time_patterns["unknown_timing"] += 1
            else:
                time_patterns["unknown_timing"] += 1
    
    hit_count = len(popular_hits)
    hit_rate = (hit_count / total_watched) * 100 if total_watched > 0 else 0
    
    # 5. 生成洞察
    insights = []
    insights.append(f"今年观看了 {total_watched} 个视频")
    insights.append(f"其中 {hit_count} 个曾经是热门视频")
    insights.append(f"热门视频命中率为 {hit_rate:.1f}%")

    # 添加观看时机洞察
    total_timed_videos = sum([time_patterns["immediate_watch"], time_patterns["trending_watch"]])
    if total_timed_videos > 0:
        immediate_rate = (time_patterns["immediate_watch"] / total_timed_videos) * 100
        if immediate_rate >= 50:
            insights.append("你是热门视频的早期发现者")
        else:
            insights.append("你喜欢在视频热门期观看")

    if hit_rate >= 50:
        insights.append("你很喜欢追热门内容！")
    elif hit_rate >= 30:
        insights.append("你对热门内容有一定关注")
    elif hit_rate >= 10:
        insights.append("你更偏爱小众内容")
    else:
        insights.append("你是真正的小众爱好者！")
    
    # 6. 关闭热门视频数据库连接
    for conn in popular_connections.values():
        if conn:
            conn.close()
    
    return {
        "total_watched": total_watched,
        "popular_hit_count": hit_count,
        "hit_rate": round(hit_rate, 2),
        "popular_videos": popular_hits[:10],  # 只返回前10个热门视频
        "time_pattern_analysis": time_patterns,
        "insights": insights
    }

def analyze_popular_prediction_ability(cursor, table_name: str, target_year: int) -> dict:
    """分析热门预测能力"""

    # 1. 获取用户观看的所有视频（按观看时间排序）
    cursor.execute(f"""
        SELECT DISTINCT bvid, title, author_name, view_at, duration, progress
        FROM {table_name}
        WHERE bvid IS NOT NULL AND bvid != ''
        ORDER BY view_at ASC
    """)

    user_videos = cursor.fetchall()
    total_watched = len(user_videos)

    if total_watched == 0:
        return {
            "total_watched": 0,
            "predicted_count": 0,
            "prediction_rate": 0,
            "insights": ["本年度没有观看记录"]
        }

    # 2. 获取热门视频数据库连接
    popular_connections = {}
    try:
        from scripts.popular_videos import get_multi_year_connections
        popular_connections = get_multi_year_connections()
    except Exception as e:
        print(f"获取热门视频数据库连接失败: {e}")
        return {
            "total_watched": total_watched,
            "predicted_count": 0,
            "prediction_rate": 0,
            "insights": [f"观看了 {total_watched} 个视频，但无法获取热门视频数据进行预测分析"]
        }

    # 3. 分析每个观看的视频是否后来成为热门
    predicted_videos = []

    for video in user_videos:
        bvid = video[0]
        view_timestamp = video[3]  # 用户观看时间戳

        # 在所有年份的热门视频数据库中查找该视频
        for year, conn in popular_connections.items():
            try:
                pop_cursor = conn.cursor()

                # 查询该视频在热门列表中的首次出现时间
                pop_cursor.execute("""
                    SELECT first_seen, title, highest_rank, appearances
                    FROM popular_video_tracking
                    WHERE bvid = ?
                """, (bvid,))

                tracking_result = pop_cursor.fetchone()

                if tracking_result:
                    first_seen_timestamp = tracking_result[0]
                    video_title = tracking_result[1]
                    highest_rank = tracking_result[2]
                    appearances = tracking_result[3]

                    # 如果用户观看时间早于视频首次成为热门的时间，说明预测成功
                    if view_timestamp < first_seen_timestamp:
                        # 计算预测提前时间（天数）
                        advance_days = (first_seen_timestamp - view_timestamp) / (24 * 3600)

                        predicted_videos.append({
                            "bvid": bvid,
                            "title": video_title or video[1],
                            "author": video[2],
                            "view_at": video[3],
                            "became_popular_at": first_seen_timestamp,
                            "advance_days": round(advance_days, 1),
                            "highest_rank": highest_rank,
                            "appearances": appearances
                        })
                        break  # 找到就跳出年份循环

            except Exception as e:
                print(f"查询 {year} 年热门视频跟踪数据失败: {e}")
                continue

    predicted_count = len(predicted_videos)
    prediction_rate = (predicted_count / total_watched) * 100 if total_watched > 0 else 0

    # 4. 生成洞察
    insights = []
    insights.append(f"今年观看了 {total_watched} 个视频")
    insights.append(f"其中 {predicted_count} 个后来成为了热门视频")
    insights.append(f"热门预测成功率为 {prediction_rate:.1f}%")

    if prediction_rate >= 10:
        insights.append("你有超强的慧眼识珠能力！")
    elif prediction_rate >= 5:
        insights.append("你对优质内容很有嗅觉")
    elif prediction_rate >= 2:
        insights.append("你偶尔能发现潜力视频")
    else:
        insights.append("你更专注于自己的兴趣领域")

    # 5. 计算平均提前天数
    if predicted_videos:
        avg_advance_days = sum(v["advance_days"] for v in predicted_videos) / len(predicted_videos)
        insights.append(f"平均提前 {avg_advance_days:.1f} 天发现热门视频")

    # 6. 关闭热门视频数据库连接
    for conn in popular_connections.values():
        if conn:
            conn.close()

    return {
        "total_watched": total_watched,
        "predicted_count": predicted_count,
        "prediction_rate": round(prediction_rate, 2),
        "predicted_videos": sorted(predicted_videos, key=lambda x: x["advance_days"], reverse=True)[:10],
        "insights": insights
    }

def analyze_author_popular_association(cursor, table_name: str, target_year: int) -> dict:
    """分析UP主热门关联"""

    # 1. 获取用户观看的所有UP主及其视频
    cursor.execute(f"""
        SELECT author_name, bvid, title, view_at, duration, progress
        FROM {table_name}
        WHERE bvid IS NOT NULL AND bvid != '' AND author_name IS NOT NULL AND author_name != ''
        ORDER BY author_name, view_at ASC
    """)

    user_videos = cursor.fetchall()

    if not user_videos:
        return {
            "total_authors": 0,
            "popular_authors": [],
            "author_stats": [],
            "insights": ["本年度没有观看记录"]
        }

    # 2. 按UP主分组统计
    author_videos = {}
    for video in user_videos:
        author_name = video[0]
        if author_name not in author_videos:
            author_videos[author_name] = []
        author_videos[author_name].append({
            "bvid": video[1],
            "title": video[2],
            "view_at": video[3],
            "duration": video[4],
            "progress": video[5]
        })

    # 3. 获取热门视频数据库连接
    popular_connections = {}
    try:
        from scripts.popular_videos import get_multi_year_connections
        popular_connections = get_multi_year_connections()
    except Exception as e:
        print(f"获取热门视频数据库连接失败: {e}")
        return {
            "total_authors": len(author_videos),
            "popular_authors": [],
            "author_stats": [],
            "insights": [f"观看了 {len(author_videos)} 个UP主的视频，但无法获取热门视频数据进行分析"]
        }

    # 4. 获取所有热门视频的bvid和作者信息
    popular_bvids = set()
    popular_video_authors = {}  # bvid -> author_name

    for year, conn in popular_connections.items():
        try:
            pop_cursor = conn.cursor()
            # 查询所有热门视频的bvid和作者
            pop_cursor.execute("SELECT DISTINCT bvid, owner_name FROM popular_videos WHERE bvid IS NOT NULL AND owner_name IS NOT NULL")
            for row in pop_cursor.fetchall():
                bvid, owner_name = row
                popular_bvids.add(bvid)
                popular_video_authors[bvid] = owner_name
        except Exception as e:
            print(f"查询 {year} 年热门视频数据失败: {e}")
            continue

    # 5. 分析每个UP主的热门视频产出能力
    author_stats = []

    for author_name, videos in author_videos.items():
        total_videos = len(videos)
        popular_videos = []

        # 检查该UP主的哪些视频成为了热门
        for video in videos:
            bvid = video["bvid"]
            if bvid in popular_bvids:
                popular_videos.append({
                    "bvid": bvid,
                    "title": video["title"],
                    "view_at": video["view_at"]
                })

        popular_count = len(popular_videos)
        popular_rate = (popular_count / total_videos) * 100 if total_videos > 0 else 0

        # 计算该UP主在热门视频数据库中的总热门视频数
        author_total_popular = 0
        for bvid, owner_name in popular_video_authors.items():
            if owner_name == author_name:
                author_total_popular += 1

        author_stats.append({
            "author_name": author_name,
            "total_videos_watched": total_videos,
            "popular_videos_watched": popular_count,
            "popular_rate": round(popular_rate, 2),
            "total_popular_videos": author_total_popular,
            "popular_videos": popular_videos[:5],  # 只返回前5个热门视频
            "efficiency_score": round(popular_rate * (popular_count + 1), 2)  # 综合评分
        })

    # 6. 按热门视频数量和热门率排序
    author_stats.sort(key=lambda x: (x["popular_videos_watched"], x["popular_rate"]), reverse=True)

    # 7. 筛选出热门制造机UP主（至少有1个热门视频）
    popular_authors = [author for author in author_stats if author["popular_videos_watched"] > 0]

    # 8. 生成洞察
    insights = []
    total_authors = len(author_videos)
    popular_author_count = len(popular_authors)

    insights.append(f"观看了 {total_authors} 个UP主的视频")
    insights.append(f"其中 {popular_author_count} 个UP主制作过热门视频")

    if popular_author_count > 0:
        avg_popular_rate = sum(author["popular_rate"] for author in popular_authors) / popular_author_count
        insights.append(f"热门UP主平均热门率为 {avg_popular_rate:.1f}%")

        top_author = popular_authors[0]
        insights.append(f"最强热门制造机：{top_author['author_name']}（{top_author['popular_videos_watched']}个热门视频）")
    else:
        insights.append("你关注的UP主都很小众哦")

    # 9. 关闭热门视频数据库连接
    for conn in popular_connections.values():
        if conn:
            conn.close()

    return {
        "total_authors": total_authors,
        "popular_author_count": popular_author_count,
        "popular_authors": popular_authors[:20],  # 返回前20个热门UP主
        "author_stats": author_stats[:10],  # 返回前10个UP主的详细统计
        "insights": insights
    }

def analyze_category_popular_distribution(cursor, table_name: str, target_year: int) -> dict:
    """分析热门视频分区分布"""

    # 1. 获取用户观看的所有视频
    cursor.execute(f"""
        SELECT DISTINCT bvid, title, author_name, view_at, duration, progress
        FROM {table_name}
        WHERE bvid IS NOT NULL AND bvid != ''
        ORDER BY view_at ASC
    """)

    user_videos = cursor.fetchall()
    total_watched = len(user_videos)

    if total_watched == 0:
        return {
            "total_watched": 0,
            "category_stats": [],
            "popular_categories": [],
            "insights": ["本年度没有观看记录"]
        }

    # 2. 获取热门视频数据库连接
    popular_connections = {}
    try:
        from scripts.popular_videos import get_multi_year_connections
        popular_connections = get_multi_year_connections()
    except Exception as e:
        print(f"获取热门视频数据库连接失败: {e}")
        return {
            "total_watched": total_watched,
            "category_stats": [],
            "popular_categories": [],
            "insights": [f"观看了 {total_watched} 个视频，但无法获取热门视频数据进行分区分析"]
        }

    # 3. 获取所有热门视频的bvid和分区信息
    popular_video_categories = {}  # bvid -> {"tid": tid, "tname": tname}
    popular_bvids = set()

    for year, conn in popular_connections.items():
        try:
            pop_cursor = conn.cursor()
            # 查询所有热门视频的bvid、分区ID和分区名称
            pop_cursor.execute("SELECT DISTINCT bvid, tid, tname FROM popular_videos WHERE bvid IS NOT NULL AND tid IS NOT NULL AND tname IS NOT NULL")
            for row in pop_cursor.fetchall():
                bvid, tid, tname = row
                popular_bvids.add(bvid)
                popular_video_categories[bvid] = {"tid": tid, "tname": tname}
        except Exception as e:
            print(f"查询 {year} 年热门视频数据失败: {e}")
            continue

    # 4. 统计用户观看的热门视频按分区分布
    category_stats = {}  # tname -> {"total_popular": count, "videos": []}

    for video in user_videos:
        bvid = video[0]
        if bvid in popular_bvids and bvid in popular_video_categories:
            category_info = popular_video_categories[bvid]
            tname = category_info["tname"]

            if tname not in category_stats:
                category_stats[tname] = {
                    "category_name": tname,
                    "tid": category_info["tid"],
                    "total_popular": 0,
                    "videos": []
                }

            category_stats[tname]["total_popular"] += 1
            category_stats[tname]["videos"].append({
                "bvid": bvid,
                "title": video[1],
                "author": video[2],
                "view_at": video[3]
            })

    # 5. 转换为列表并排序
    category_list = list(category_stats.values())
    category_list.sort(key=lambda x: x["total_popular"], reverse=True)

    # 6. 计算统计数据
    total_popular_watched = sum(cat["total_popular"] for cat in category_list)
    popular_rate = (total_popular_watched / total_watched) * 100 if total_watched > 0 else 0

    # 7. 生成洞察
    insights = []
    insights.append(f"今年观看了 {total_watched} 个视频")
    insights.append(f"其中 {total_popular_watched} 个曾经是热门视频")
    insights.append(f"热门视频分布在 {len(category_list)} 个分区")

    if category_list:
        top_category = category_list[0]
        insights.append(f"最爱的热门分区：{top_category['category_name']}（{top_category['total_popular']}个热门视频）")

        # 分析分区偏好
        if len(category_list) >= 3:
            top_3_count = sum(cat["total_popular"] for cat in category_list[:3])
            top_3_rate = (top_3_count / total_popular_watched) * 100 if total_popular_watched > 0 else 0
            if top_3_rate >= 70:
                insights.append("你的热门视频偏好很集中")
            else:
                insights.append("你的热门视频偏好很多样化")

        # 分析热门敏感度
        if popular_rate >= 50:
            insights.append("你对各分区的热门内容都很敏感")
        elif popular_rate >= 30:
            insights.append("你在某些分区有不错的热门嗅觉")
        else:
            insights.append("你更专注于特定分区的小众内容")

    # 8. 关闭热门视频数据库连接
    for conn in popular_connections.values():
        if conn:
            conn.close()

    return {
        "total_watched": total_watched,
        "total_popular_watched": total_popular_watched,
        "popular_rate": round(popular_rate, 2),
        "category_count": len(category_list),
        "category_stats": category_list,
        "popular_categories": category_list[:10],  # 返回前10个热门分区
        "insights": insights
    }

def analyze_duration_popular_distribution(cursor, table_name: str, target_year: int) -> dict:
    """分析热门视频时长分布"""

    # 1. 获取用户观看的所有视频
    cursor.execute(f"""
        SELECT DISTINCT bvid, title, author_name, view_at, duration, progress
        FROM {table_name}
        WHERE bvid IS NOT NULL AND bvid != '' AND duration IS NOT NULL AND duration > 0
        ORDER BY view_at ASC
    """)

    user_videos = cursor.fetchall()
    total_watched = len(user_videos)

    if total_watched == 0:
        return {
            "total_watched": 0,
            "duration_stats": [],
            "popular_duration_videos": [],
            "insights": ["本年度没有观看记录"]
        }

    # 2. 获取热门视频数据库连接
    popular_connections = {}
    try:
        from scripts.popular_videos import get_multi_year_connections
        popular_connections = get_multi_year_connections()
    except Exception as e:
        print(f"获取热门视频数据库连接失败: {e}")
        return {
            "total_watched": total_watched,
            "duration_stats": [],
            "popular_duration_videos": [],
            "insights": [f"观看了 {total_watched} 个视频，但无法获取热门视频数据进行时长分析"]
        }

    # 3. 获取所有热门视频的bvid和时长信息
    popular_video_durations = {}  # bvid -> duration
    popular_bvids = set()

    for year, conn in popular_connections.items():
        try:
            pop_cursor = conn.cursor()
            # 查询所有热门视频的bvid和时长
            pop_cursor.execute("SELECT DISTINCT bvid, duration FROM popular_videos WHERE bvid IS NOT NULL AND duration IS NOT NULL AND duration > 0")
            for row in pop_cursor.fetchall():
                bvid, duration = row
                popular_bvids.add(bvid)
                popular_video_durations[bvid] = duration
        except Exception as e:
            print(f"查询 {year} 年热门视频数据失败: {e}")
            continue

    # 4. 定义时长区间（秒）
    duration_ranges = {
        "短视频": {"min": 0, "max": 300, "videos": [], "count": 0},  # ≤5分钟
        "中等视频": {"min": 300, "max": 1200, "videos": [], "count": 0},  # 5-20分钟
        "长视频": {"min": 1200, "max": 3600, "videos": [], "count": 0},  # 20-60分钟
        "超长视频": {"min": 3600, "max": float('inf'), "videos": [], "count": 0}  # >60分钟
    }

    # 5. 统计用户观看的热门视频按时长分布
    total_popular_watched = 0

    for video in user_videos:
        bvid = video[0]
        user_duration = video[4]  # 用户历史记录中的时长

        if bvid in popular_bvids:
            total_popular_watched += 1
            # 使用热门视频数据库中的时长，如果没有则使用用户记录中的时长
            duration = popular_video_durations.get(bvid, user_duration)

            # 分类到对应的时长区间
            for range_name, range_info in duration_ranges.items():
                if range_info["min"] <= duration < range_info["max"]:
                    range_info["count"] += 1
                    range_info["videos"].append({
                        "bvid": bvid,
                        "title": video[1],
                        "author": video[2],
                        "view_at": video[3],
                        "duration": duration,
                        "formatted_duration": format_duration(duration)
                    })
                    break

    # 6. 计算统计数据
    popular_rate = (total_popular_watched / total_watched) * 100 if total_watched > 0 else 0

    # 7. 生成洞察
    insights = []
    insights.append(f"今年观看了 {total_watched} 个视频")
    insights.append(f"其中 {total_popular_watched} 个曾经是热门视频")

    if total_popular_watched > 0:
        # 找出最偏爱的时长类型
        max_count = 0
        favorite_duration_type = ""
        for range_name, range_info in duration_ranges.items():
            if range_info["count"] > max_count:
                max_count = range_info["count"]
                favorite_duration_type = range_name

        if favorite_duration_type:
            favorite_rate = (max_count / total_popular_watched) * 100
            insights.append(f"最偏爱{favorite_duration_type}热门内容（{max_count}个，占{favorite_rate:.1f}%）")

        # 分析时长偏好特征
        short_count = duration_ranges["短视频"]["count"]
        medium_count = duration_ranges["中等视频"]["count"]
        long_count = duration_ranges["长视频"]["count"]
        super_long_count = duration_ranges["超长视频"]["count"]

        if short_count >= total_popular_watched * 0.5:
            insights.append("你偏爱快节奏的短视频内容")
        elif long_count + super_long_count >= total_popular_watched * 0.5:
            insights.append("你喜欢深度的长视频内容")
        elif medium_count >= total_popular_watched * 0.4:
            insights.append("你偏爱适中时长的视频内容")
        else:
            insights.append("你对各种时长的视频都有涉猎")

    # 8. 准备返回数据
    duration_stats = []
    for range_name, range_info in duration_ranges.items():
        if range_info["count"] > 0:
            duration_stats.append({
                "duration_type": range_name,
                "count": range_info["count"],
                "percentage": round((range_info["count"] / total_popular_watched) * 100, 1) if total_popular_watched > 0 else 0,
                "videos": sorted(range_info["videos"], key=lambda x: x["view_at"], reverse=True)[:5]  # 最近观看的5个
            })

    # 按数量排序
    duration_stats.sort(key=lambda x: x["count"], reverse=True)

    # 9. 关闭热门视频数据库连接
    for conn in popular_connections.values():
        if conn:
            conn.close()

    return {
        "total_watched": total_watched,
        "total_popular_watched": total_popular_watched,
        "popular_rate": round(popular_rate, 2),
        "duration_stats": duration_stats,
        "popular_duration_videos": duration_stats[:4],  # 返回前4个时长类型
        "insights": insights
    }

def format_duration(seconds):
    """格式化时长显示"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒" if secs > 0 else f"{minutes}分钟"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟" if minutes > 0 else f"{hours}小时"



@router.get("/popular-hit-rate", summary="获取热门视频命中率分析")
async def get_popular_hit_rate(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取热门视频命中率分析
    
    分析用户观看的视频中有多少曾经是热门视频
    
    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据
    
    Returns:
        dict: 包含热门视频命中率分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应
    
    # 检查缓存
    if use_cache:
        try:
            from .title_pattern_discovery import pattern_cache
            cached_data = pattern_cache.get_cached_patterns(table_name, 'popular_hit_rate')
            if cached_data:
                print(f"使用 {target_year} 年的热门命中率分析缓存数据")
                return cached_data
        except Exception as e:
            print(f"获取缓存失败: {e}")
    
    conn = None
    try:
        # 连接数据库
        conn = get_db()
        cursor = conn.cursor()
        
        # 分析热门视频命中率
        hit_rate_analysis = analyze_popular_hit_rate(cursor, table_name, target_year)
        
        # 构建响应
        response = {
            "status": "success",
            "data": {
                "hit_rate_analysis": hit_rate_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }
        
        # 更新缓存
        if use_cache:
            try:
                from .title_pattern_discovery import pattern_cache
                print(f"更新 {target_year} 年的热门命中率分析数据缓存")
                pattern_cache.cache_patterns(table_name, 'popular_hit_rate', response)
            except Exception as e:
                print(f"更新缓存失败: {e}")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/popular-prediction-ability", summary="获取热门预测能力分析")
async def get_popular_prediction_ability(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取热门预测能力分析

    分析用户观看的视频中，有多少后来成为了热门视频，评估用户的"慧眼识珠"能力

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含热门预测能力分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    # 检查缓存
    if use_cache:
        try:
            from .title_pattern_discovery import pattern_cache
            cached_data = pattern_cache.get_cached_patterns(table_name, 'popular_prediction_ability')
            if cached_data:
                print(f"使用 {target_year} 年的热门预测能力分析缓存数据")
                return cached_data
        except Exception as e:
            print(f"获取缓存失败: {e}")

    conn = None
    try:
        # 连接数据库
        conn = get_db()
        cursor = conn.cursor()

        # 分析热门预测能力
        prediction_analysis = analyze_popular_prediction_ability(cursor, table_name, target_year)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "prediction_analysis": prediction_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        if use_cache:
            try:
                from .title_pattern_discovery import pattern_cache
                print(f"更新 {target_year} 年的热门预测能力分析数据缓存")
                pattern_cache.cache_patterns(table_name, 'popular_prediction_ability', response)
            except Exception as e:
                print(f"更新缓存失败: {e}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/author-popular-association", summary="获取UP主热门关联分析")
async def get_author_popular_association(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取UP主热门关联分析

    分析关注UP主的热门视频产出能力，统计"热门制造机"UP主

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含UP主热门关联分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    # 检查缓存
    if use_cache:
        try:
            from .title_pattern_discovery import pattern_cache
            cached_data = pattern_cache.get_cached_patterns(table_name, 'author_popular_association')
            if cached_data:
                print(f"使用 {target_year} 年的UP主热门关联分析缓存数据")
                return cached_data
        except Exception as e:
            print(f"获取缓存失败: {e}")

    conn = None
    try:
        # 连接数据库
        conn = get_db()
        cursor = conn.cursor()

        # 分析UP主热门关联
        association_analysis = analyze_author_popular_association(cursor, table_name, target_year)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "association_analysis": association_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        if use_cache:
            try:
                from .title_pattern_discovery import pattern_cache
                print(f"更新 {target_year} 年的UP主热门关联分析数据缓存")
                pattern_cache.cache_patterns(table_name, 'author_popular_association', response)
            except Exception as e:
                print(f"更新缓存失败: {e}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/category-popular-distribution", summary="获取热门视频分区分布分析")
async def get_category_popular_distribution(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取热门视频分区分布分析

    分析用户观看的热门视频在各个分区的分布情况

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含热门视频分区分布分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    # 检查缓存
    if use_cache:
        try:
            from .title_pattern_discovery import pattern_cache
            cached_data = pattern_cache.get_cached_patterns(table_name, 'category_popular_distribution')
            if cached_data:
                print(f"使用 {target_year} 年的热门视频分区分布分析缓存数据")
                return cached_data
        except Exception as e:
            print(f"获取缓存失败: {e}")

    conn = None
    try:
        # 连接数据库
        conn = get_db()
        cursor = conn.cursor()

        # 分析热门视频分区分布
        distribution_analysis = analyze_category_popular_distribution(cursor, table_name, target_year)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "distribution_analysis": distribution_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        if use_cache:
            try:
                from .title_pattern_discovery import pattern_cache
                print(f"更新 {target_year} 年的热门视频分区分布分析数据缓存")
                pattern_cache.cache_patterns(table_name, 'category_popular_distribution', response)
            except Exception as e:
                print(f"更新缓存失败: {e}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/duration-popular-distribution", summary="获取热门视频时长分布分析")
async def get_duration_popular_distribution(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取热门视频时长分布分析

    分析用户观看的热门视频在不同时长区间的分布情况

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含热门视频时长分布分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    # 检查缓存
    if use_cache:
        try:
            from .title_pattern_discovery import pattern_cache
            cached_data = pattern_cache.get_cached_patterns(table_name, 'duration_popular_distribution')
            if cached_data:
                print(f"使用 {target_year} 年的热门视频时长分布分析缓存数据")
                return cached_data
        except Exception as e:
            print(f"获取缓存失败: {e}")

    conn = None
    try:
        # 连接数据库
        conn = get_db()
        cursor = conn.cursor()

        # 分析热门视频时长分布
        duration_analysis = analyze_duration_popular_distribution(cursor, table_name, target_year)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "duration_analysis": duration_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        if use_cache:
            try:
                from .title_pattern_discovery import pattern_cache
                print(f"更新 {target_year} 年的热门视频时长分布分析数据缓存")
                pattern_cache.cache_patterns(table_name, 'duration_popular_distribution', response)
            except Exception as e:
                print(f"更新缓存失败: {e}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

