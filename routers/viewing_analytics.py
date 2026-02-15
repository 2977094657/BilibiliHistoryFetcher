import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from scripts.utils import load_config, get_output_path

router = APIRouter()
config = load_config()

def get_db():
    """获取数据库连接"""
    db_path = get_output_path(config['db_file'])
    return sqlite3.connect(db_path)

def generate_continuity_insights(continuity_data: dict) -> dict:
    """生成连续性相关的洞察"""
    insights = {}
    
    # 连续性洞察
    max_streak = continuity_data['max_streak']
    current_streak = continuity_data['current_streak']
    
    # 根据连续天数生成评价
    streak_comment = ""
    if max_streak >= 30:
        streak_comment = "看来是B站的重度使用者"
    elif max_streak >= 14:
        streak_comment = "看来你对B站情有独钟呢！"
    else:
        streak_comment = "你的观看习惯比较随性自由。"
    
    insights['continuity'] = f"你最长连续观看B站达到了{max_streak}天，{streak_comment}目前已经连续观看了{current_streak}天。"
    
    return insights

def analyze_viewing_continuity(cursor, table_name: str) -> dict:
    """分析观看习惯的连续性
    
    Args:
        cursor: 数据库游标
        table_name: 表名
    
    Returns:
        dict: 连续性分析结果
    """
    # 获取所有观看日期
    cursor.execute(f"""
        SELECT DISTINCT date(datetime(view_at + 28800, 'unixepoch')) as view_date
        FROM {table_name}
        ORDER BY view_date
    """)
    dates = [row[0] for row in cursor.fetchall()]
    
    # 计算连续观看天数
    max_streak = current_streak = 1
    longest_streak_start = longest_streak_end = current_streak_start = dates[0] if dates else None
    
    for i in range(1, len(dates)):
        date1 = datetime.strptime(dates[i-1], '%Y-%m-%d')
        date2 = datetime.strptime(dates[i], '%Y-%m-%d')
        if (date2 - date1).days == 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                longest_streak_start = datetime.strptime(dates[i-max_streak+1], '%Y-%m-%d').strftime('%Y-%m-%d')
                longest_streak_end = dates[i]
        else:
            current_streak = 1
            current_streak_start = dates[i]
    
    return {
        'max_streak': max_streak,
        'longest_streak_period': {
            'start': longest_streak_start,
            'end': longest_streak_end
        },
        'current_streak': current_streak,
        'current_streak_start': current_streak_start
    }

def analyze_time_investment(cursor, table_name: str) -> dict:
    """分析时间投入强度
    
    Args:
        cursor: 数据库游标
        table_name: 表名
    
    Returns:
        dict: 时间投入分析结果
    """
    cursor.execute(f"""
        SELECT 
            date(datetime(view_at + 28800, 'unixepoch')) as view_date,
            COUNT(*) as video_count,
            SUM(CASE WHEN progress = -1 THEN duration ELSE progress END) as total_duration
        FROM {table_name}
        GROUP BY view_date
        ORDER BY total_duration DESC
        LIMIT 1
    """)
    max_duration_day = cursor.fetchone()
    
    cursor.execute(f"""
        SELECT 
            AVG(daily_duration) as avg_duration
        FROM (
            SELECT 
                date(datetime(view_at + 28800, 'unixepoch')) as view_date,
                SUM(CASE WHEN progress = -1 THEN duration ELSE progress END) as daily_duration
            FROM {table_name}
            GROUP BY view_date
        )
    """)
    avg_daily_duration = cursor.fetchone()[0]
    
    return {
        'max_duration_day': {
            'date': max_duration_day[0],
            'video_count': max_duration_day[1],
            'total_duration': max_duration_day[2]
        },
        'avg_daily_duration': avg_daily_duration
    }

def analyze_completion_rates(cursor, table_name: str) -> dict:
    """分析视频完成率"""
    # 获取表结构
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1]: idx for idx, col in enumerate(cursor.fetchall())}
    
    # 确保必要的列存在
    required_columns = ['duration', 'progress', 'author_name', 'author_mid', 'tag_name']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Required column '{col}' not found in table {table_name}")
    
    cursor.execute(f"SELECT * FROM {table_name}")
    histories = cursor.fetchall()
    
    # 基础统计
    total_videos = len(histories)
    total_completion = 0
    fully_watched = 0
    not_started = 0
    
    # UP主统计
    author_stats = {}
    # 分区统计
    tag_stats = {}
    # 时长分布统计
    duration_stats = {
        "短视频(≤5分钟)": {"video_count": 0, "total_completion": 0, "fully_watched": 0, "average_completion_rate": 0},
        "中等视频(5-20分钟)": {"video_count": 0, "total_completion": 0, "fully_watched": 0, "average_completion_rate": 0},
        "长视频(>20分钟)": {"video_count": 0, "total_completion": 0, "fully_watched": 0, "average_completion_rate": 0}
    }
    # 完成率分布
    completion_distribution = {
        "0-10%": 0,
        "10-30%": 0,
        "30-50%": 0,
        "50-70%": 0,
        "70-90%": 0,
        "90-100%": 0
    }
    
    for history in histories:
        # 获取并转换数据类型
        try:
            duration = float(history[columns['duration']]) if history[columns['duration']] else 0
            progress = float(history[columns['progress']]) if history[columns['progress']] else 0
            author_name = history[columns['author_name']]
            author_mid = history[columns['author_mid']]
            tag_name = history[columns['tag_name']]
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to process record: {history}")
            continue
        
        # 计算完成率
        # 当progress为-1时表示已完全观看，计算为100%
        if progress == -1:
            completion_rate = 100
        else:
            completion_rate = (progress / duration * 100) if duration > 0 else 0
        
        total_completion += completion_rate
        
        # 统计完整观看和未开始观看
        if completion_rate >= 90:  # 90%以上视为完整观看
            fully_watched += 1
        elif completion_rate == 0:
            not_started += 1
        
        # 更新完成率分布
        if completion_rate <= 10:
            completion_distribution["0-10%"] += 1
        elif completion_rate <= 30:
            completion_distribution["10-30%"] += 1
        elif completion_rate <= 50:
            completion_distribution["30-50%"] += 1
        elif completion_rate <= 70:
            completion_distribution["50-70%"] += 1
        elif completion_rate <= 90:
            completion_distribution["70-90%"] += 1
        else:
            completion_distribution["90-100%"] += 1
        
        # UP主统计
        if author_name and author_mid:
            if author_name not in author_stats:
                author_stats[author_name] = {
                    "author_mid": author_mid,
                    "video_count": 0,
                    "total_completion": 0,
                    "fully_watched": 0
                }
            stats = author_stats[author_name]
            stats["video_count"] += 1
            stats["total_completion"] += completion_rate
            if completion_rate >= 90:
                stats["fully_watched"] += 1
        
        # 分区统计
        if tag_name:
            if tag_name not in tag_stats:
                tag_stats[tag_name] = {
                    "video_count": 0,
                    "total_completion": 0,
                    "fully_watched": 0
                }
            stats = tag_stats[tag_name]
            stats["video_count"] += 1
            stats["total_completion"] += completion_rate
            if completion_rate >= 90:
                stats["fully_watched"] += 1
        
        # 时长分布统计
        if duration <= 300:  # 5分钟
            category = "短视频(≤5分钟)"
        elif duration <= 1200:  # 20分钟
            category = "中等视频(5-20分钟)"
        else:
            category = "长视频(>20分钟)"
        
        stats = duration_stats[category]
        stats["video_count"] += 1
        stats["total_completion"] += completion_rate
        if completion_rate >= 90:
            stats["fully_watched"] += 1
    
    # 计算总体统计
    overall_stats = {
        "total_videos": total_videos,
        "average_completion_rate": round(total_completion / total_videos, 2) if total_videos > 0 else 0,
        "fully_watched_count": fully_watched,
        "not_started_count": not_started,
        "fully_watched_rate": round(fully_watched / total_videos * 100, 2) if total_videos > 0 else 0,
        "not_started_rate": round(not_started / total_videos * 100, 2) if total_videos > 0 else 0
    }
    
    # 计算各类视频的平均完成率和完整观看率
    for category, stats in duration_stats.items():
        if stats["video_count"] > 0:
            stats["average_completion_rate"] = round(stats["total_completion"] / stats["video_count"], 2)
            stats["fully_watched_rate"] = round(stats["fully_watched"] / stats["video_count"] * 100, 2)
        else:
            stats["average_completion_rate"] = 0
            stats["fully_watched_rate"] = 0
    
    # 计算UP主平均完成率和完整观看率，并按观看数量筛选和排序
    filtered_authors = {}
    for name, stats in author_stats.items():
        if stats["video_count"] >= 5:  # 只保留观看数量>=5的UP主
            stats["average_completion_rate"] = round(stats["total_completion"] / stats["video_count"], 2)
            stats["fully_watched_rate"] = round(stats["fully_watched"] / stats["video_count"] * 100, 2)
            filtered_authors[name] = stats

    most_watched_authors = {}
    highest_completion_authors = {}
    
    # 计算分区平均完成率和完整观看率，并按观看数量筛选和排序
    filtered_tags = {}
    for tag, stats in tag_stats.items():
        if stats["video_count"] >= 5:  # 只保留视频数量>=5的分区
            stats["average_completion_rate"] = round(stats["total_completion"] / stats["video_count"], 2)
            stats["fully_watched_rate"] = round(stats["fully_watched"] / stats["video_count"] * 100, 2)
            filtered_tags[tag] = stats

    top_tags = {}
    
    return {
        "overall_stats": overall_stats,
        "duration_based_stats": duration_stats,
        "completion_distribution": completion_distribution,
        "tag_completion_rates": top_tags,
        "most_watched_authors": most_watched_authors,
        "highest_completion_authors": highest_completion_authors
    }

def generate_completion_insights(completion_data: dict) -> dict:
    """生成视频完成率相关的洞察"""
    insights = {}
    
    try:
        # 整体完成率洞察
        overall = completion_data.get("overall_stats", {})
        if overall:
            insights["overall_completion"] = (
                f"在观看的{overall.get('total_videos', 0)}个视频中，你平均观看完成率为{overall.get('average_completion_rate', 0)}%，"
                f"完整看完的视频占比{overall.get('fully_watched_rate', 0)}%，未开始观看的占比{overall.get('not_started_rate', 0)}%。"
            )
        
        # 视频时长偏好洞察
        duration_stats = completion_data.get("duration_based_stats", {})
        if duration_stats:
            valid_durations = [(k, v) for k, v in duration_stats.items() 
                             if v.get("video_count", 0) > 0 and v.get("average_completion_rate", 0) > 0]
            if valid_durations:
                max_completion_duration = max(valid_durations, key=lambda x: x[1].get("average_completion_rate", 0))
                insights["duration_preference"] = (
                    f"你最容易看完的是{max_completion_duration[0]}，平均完成率达到{max_completion_duration[1].get('average_completion_rate', 0)}%，"
                    f"其中完整看完的视频占比{max_completion_duration[1].get('fully_watched_rate', 0)}%。"
                )
        
        # 分区兴趣洞察
        tag_rates = completion_data.get("tag_completion_rates", {})
        if tag_rates:
            valid_tags = [(k, v) for k, v in tag_rates.items() 
                         if v.get("video_count", 0) >= 5]  # 只考虑观看数量>=5的分区
            if valid_tags:
                top_tag = max(valid_tags, key=lambda x: x[1].get("average_completion_rate", 0))
                insights["tag_completion"] = (
                    f"在经常观看的分区中，你对{top_tag[0]}分区的视频最感兴趣，平均完成率达到{top_tag[1].get('average_completion_rate', 0)}%，"
                    f"观看过{top_tag[1].get('video_count', 0)}个该分区的视频。"
                )
        
        # UP主偏好洞察
        most_watched = completion_data.get("most_watched_authors", {})
        if most_watched:
            top_watched = next(iter(most_watched.items()), None)
            if top_watched:
                insights["most_watched_author"] = (
                    f"你观看最多的UP主是{top_watched[0]}，观看了{top_watched[1].get('video_count', 0)}个视频，"
                    f"平均完成率为{top_watched[1].get('average_completion_rate', 0)}%。"
                )
        
        highest_completion = completion_data.get("highest_completion_authors", {})
        if highest_completion and most_watched:
            top_completion = next(iter(highest_completion.items()), None)
            if top_completion and top_completion[0] != next(iter(most_watched.keys())):
                insights["highest_completion_author"] = (
                    f"在经常观看的UP主中，你对{top_completion[0]}的视频完成度最高，"
                    f"平均完成率达到{top_completion[1].get('average_completion_rate', 0)}%，"
                    f"观看过{top_completion[1].get('video_count', 0)}个视频。"
                )
    
    except Exception as e:
        print(f"Error generating completion insights: {str(e)}")
        # 返回一个基础的洞察信息
        insights["basic_completion"] = "暂时无法生成详细的观看完成率分析。"
    
    return insights

def analyze_video_watch_counts(cursor, table_name: str) -> dict:
    """分析视频观看次数"""
    # 获取表结构
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1]: idx for idx, col in enumerate(cursor.fetchall())}
    
    # 确保必要的列存在
    required_columns = ['title', 'bvid', 'duration', 'tag_name', 'author_name']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Required column '{col}' not found in table {table_name}")
    
    # 获取视频观看次数统计
    cursor.execute(f"""
        SELECT 
            title,
            bvid,
            duration,
            tag_name,
            author_name,
            COUNT(*) as watch_count,
            MIN(view_at) as first_view,
            MAX(view_at) as last_view
        FROM {table_name}
        WHERE bvid IS NOT NULL AND bvid != ''
        GROUP BY bvid
        HAVING COUNT(*) > 1
        ORDER BY watch_count DESC
    """)
    
    results = cursor.fetchall()
    
    # 处理统计结果
    most_watched_videos = []
    total_rewatched = 0
    total_videos = len(results)
    duration_distribution = {
        "短视频(≤5分钟)": 0,
        "中等视频(5-20分钟)": 0,
        "长视频(>20分钟)": 0
    }
    tag_distribution = {}
    
    for row in results:
        title = row[0]
        bvid = row[1]
        duration = float(row[2]) if row[2] else 0
        tag_name = row[3]
        author_name = row[4]
        watch_count = row[5]
        first_view = row[6]
        last_view = row[7]
        
        # 统计重复观看的视频时长分布
        if duration <= 300:
            duration_distribution["短视频(≤5分钟)"] += 1
        elif duration <= 1200:
            duration_distribution["中等视频(5-20分钟)"] += 1
        else:
            duration_distribution["长视频(>20分钟)"] += 1

        
        # 记录观看次数最多的视频
        if len(most_watched_videos) < 10:
            most_watched_videos.append({
                "title": title,
                "bvid": bvid,
                "duration": duration,
                "tag_name": tag_name,
                "author_name": author_name,
                "watch_count": watch_count,
                "first_view": first_view,
                "last_view": last_view,
                "avg_interval": (last_view - first_view) / (watch_count - 1) if watch_count > 1 else 0
            })
        
        total_rewatched += watch_count - 1
    
    # 获取总视频数
    cursor.execute(f"SELECT COUNT(DISTINCT bvid) FROM {table_name}")
    total_unique_videos = cursor.fetchone()[0]
    
    # 计算重复观看率
    rewatch_rate = round(total_videos / total_unique_videos * 100, 2)
    
    # 获取分区排名
    tag_ranking = sorted(
        tag_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return {
        "rewatch_stats": {
            "total_rewatched_videos": total_videos,
            "total_unique_videos": total_unique_videos,
            "rewatch_rate": rewatch_rate,
            "total_rewatch_count": total_rewatched
        },
        "most_watched_videos": most_watched_videos,
        "duration_distribution": duration_distribution,
        "tag_distribution": dict(tag_ranking)
    }

def generate_watch_count_insights(watch_count_data: dict) -> dict:
    """生成视频观看次数相关的洞察"""
    insights = {}
    
    try:
        # 重复观看统计洞察
        rewatch_stats = watch_count_data.get("rewatch_stats", {})
        total_videos = rewatch_stats.get('total_unique_videos', 0)
        rewatched_videos = rewatch_stats.get('total_rewatched_videos', 0)
        rewatch_rate = rewatch_stats.get('rewatch_rate', 0)
        total_rewatches = rewatch_stats.get('total_rewatch_count', 0)
        
        if rewatched_videos > 0:
            avg_watches_per_video = round(total_rewatches/rewatched_videos + 1, 1)
            insights["rewatch_overview"] = (
                f"在你观看的{total_videos}个视频中，有{rewatched_videos}个视频被重复观看，"
                f"重复观看率为{rewatch_rate}%。这些视频总共被额外观看了{total_rewatches}次，"
                f"平均每个重复观看的视频被看了{avg_watches_per_video}次。"
            )
        else:
            insights["rewatch_overview"] = f"在你观看的{total_videos}个视频中，暂时还没有重复观看的视频。"
        
        # 最多观看视频洞察
        most_watched = watch_count_data.get("most_watched_videos", [])
        if most_watched:
            top_videos = most_watched[:3]  # 取前三
            top_video = top_videos[0]
            
            # 计算平均重看间隔（天）
            avg_interval_days = round(top_video.get('avg_interval', 0) / (24 * 3600), 1)
            
            if avg_interval_days > 0:
                insights["most_watched_videos"] = (
                    f"你最喜欢的视频是{top_video.get('author_name', '未知作者')}的《{top_video.get('title', '未知视频')}》，"
                    f"共观看了{top_video.get('watch_count', 0)}次，平均每{avg_interval_days}天就会重看一次。"
                )
            else:
                insights["most_watched_videos"] = (
                    f"你最喜欢的视频是{top_video.get('author_name', '未知作者')}的《{top_video.get('title', '未知视频')}》，"
                    f"共观看了{top_video.get('watch_count', 0)}次。"
                )
            
            if len(top_videos) > 1:
                other_favorites = [
                    f"{v.get('author_name', '未知作者')}的《{v.get('title', '未知视频')}》({v.get('watch_count', 0)}次)"
                    for v in top_videos[1:3]
                ]
                insights["most_watched_videos"] += f"紧随其后的是{' 和 '.join(other_favorites)}。"
        
        # 重复观看视频时长分布洞察
        duration_dist = watch_count_data.get("duration_distribution", {})
        if duration_dist:
            total_rewatched = sum(duration_dist.values())
            if total_rewatched > 0:
                duration_percentages = {
                    k: round(v/total_rewatched * 100, 1)
                    for k, v in duration_dist.items()
                }
                sorted_durations = sorted(
                    duration_percentages.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                insights["duration_preference"] = (
                    f"在重复观看的视频中，{sorted_durations[0][0]}最多，占比{sorted_durations[0][1]}%。"
                    f"其次是{sorted_durations[1][0]}({sorted_durations[1][1]}%)，"
                    f"而{sorted_durations[2][0]}占比{sorted_durations[2][1]}%。"
                    f"这表明你在重复观看时更偏好{sorted_durations[0][0].replace('视频', '')}的内容。"
                )
        
        # 重复观看分区分布洞察
        tag_dist = watch_count_data.get("tag_distribution", {})
        if tag_dist:
            total_tags = sum(tag_dist.values())
            if total_tags > 0:
                top_tags = sorted(tag_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                
                tag_insights = []
                for tag, count in top_tags:
                    percentage = round(count/total_tags * 100, 1)
                    tag_insights.append(f"{tag}({count}个视频, {percentage}%)")
                
                insights["tag_preference"] = (
                    f"你最常重复观看的内容类型是{tag_insights[0]}。"
                    f"紧随其后的是{' 和 '.join(tag_insights[1:])}。"
                )
        
        # 生成总体观看行为总结
        if rewatched_videos > 0:
            insights["behavior_summary"] = (
                f"总的来说，你是一位{_get_rewatch_habit_description(rewatch_rate)}。"
                f"你特别喜欢重复观看{_get_preferred_content_type(tag_dist, duration_dist)}的内容。"
            )
        else:
            insights["behavior_summary"] = "总的来说，你喜欢探索新的内容，很少重复观看同一个视频。"
    
    except Exception as e:
        print(f"Error generating watch count insights: {str(e)}")
        insights["basic_watch_count"] = "暂时无法生成详细的重复观看分析。"
    
    return insights

def _get_rewatch_habit_description(rewatch_rate: float) -> str:
    """根据重复观看率描述用户习惯"""
    if rewatch_rate < 2:
        return "喜欢探索新内容的观众"
    elif rewatch_rate < 5:
        return "对特定内容会重复观看的观众"
    else:
        return "经常重复观看喜欢内容的忠实观众"

def _get_preferred_content_type(tag_dist: dict, duration_dist: dict) -> str:
    """根据分区和时长分布描述用户偏好"""
    if not tag_dist or not duration_dist:
        return "多样化"
        
    top_tag = max(tag_dist.items(), key=lambda x: x[1])[0]
    top_duration = max(duration_dist.items(), key=lambda x: x[1])[0]
    
    return f"{top_duration.replace('视频', '')}的{top_tag}"

def get_available_years():
    """获取数据库中所有可用的年份"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'bilibili_history_%'
            ORDER BY name DESC
        """)

        years = []
        for (table_name,) in cursor.fetchall():
            try:
                year = int(table_name.split('_')[-1])
                years.append(year)
            except (ValueError, IndexError):
                continue

        return sorted(years, reverse=True)
    except sqlite3.Error as e:
        print(f"获取年份列表时发生错误: {e}")
        return []
    finally:
        if conn:
            conn.close()

def validate_year_and_get_table(year: Optional[int]) -> tuple:
    """验证年份并返回表名和可用年份列表

    Args:
        year: 要验证的年份，None表示使用最新年份

    Returns:
        tuple: (table_name, target_year, available_years) 或 (None, None, error_response)
    """
    # 获取可用年份列表
    available_years = get_available_years()
    if not available_years:
        error_response = {
            "status": "error",
            "message": "未找到任何历史记录数据"
        }
        return None, None, error_response

    # 如果未指定年份，使用最新的年份；如果指定年份不存在，则自动回退到最新可用年份
    if year is None:
        target_year = available_years[0]
    elif year in available_years:
        target_year = year
    else:
        target_year = available_years[0]

    table_name = f"bilibili_history_{target_year}"
    return table_name, target_year, available_years

@router.get("/monthly-stats", summary="获取月度观看统计分析")
async def get_monthly_stats(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取月度观看统计分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含月度观看统计分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'monthly_stats')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的月度统计分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的月度观看统计数据")

        # 月度观看统计
        cursor.execute(f"""
            SELECT
                strftime('%Y-%m', datetime(view_at + 28800, 'unixepoch')) as month,
                COUNT(*) as view_count
            FROM {table_name}
            GROUP BY month
            ORDER BY month
        """)
        monthly_stats = {row[0]: row[1] for row in cursor.fetchall()}

        # 计算总视频数和活跃天数
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_videos = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT COUNT(DISTINCT strftime('%Y-%m-%d', datetime(view_at + 28800, 'unixepoch')))
            FROM {table_name}
        """)
        active_days = cursor.fetchone()[0]

        # 计算平均每日观看数
        avg_daily_videos = round(total_videos / active_days, 1) if active_days > 0 else 0

        # 生成月度洞察
        monthly_insights = {}

        # 添加总体活动洞察
        if total_videos > 0 and active_days > 0:
            monthly_insights['overall_activity'] = f"今年以来，你在B站观看了{total_videos}个视频，平均每天观看{avg_daily_videos}个视频"

        if monthly_stats:
            max_month = max(monthly_stats.items(), key=lambda x: x[1])
            min_month = min(monthly_stats.items(), key=lambda x: x[1])

            # 计算月度趋势
            months = sorted(monthly_stats.keys())
            month_trend = ""
            if len(months) >= 2:
                first_month_count = monthly_stats[months[0]]
                last_month_count = monthly_stats[months[-1]]
                if last_month_count > first_month_count * 1.2:
                    month_trend = "你在B站观看视频的热情正在逐月增长，看来你越来越喜欢B站了呢！"
                elif last_month_count < first_month_count * 0.8:
                    month_trend = "最近你在B站的活跃度有所下降，可能是工作或学习变得更忙了吧。"
                else:
                    month_trend = "你在B站的活跃度保持得很稳定，看来已经养成了良好的观看习惯。"

            monthly_insights['monthly_pattern'] = f"在{max_month[0]}月，你观看了{max_month[1]}个视频，是你最活跃的月份；而在{min_month[0]}月，观看量为{min_month[1]}个。{month_trend}"

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "monthly_stats": monthly_stats,
                "total_videos": total_videos,
                "active_days": active_days,
                "avg_daily_videos": avg_daily_videos,
                "insights": monthly_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的月度统计分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'monthly_stats', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/weekly-stats", summary="获取周度观看统计分析")
async def get_weekly_stats(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取周度观看统计分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含周度观看统计分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'weekly_stats')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的周度统计分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的周度观看统计数据")

        # 计算活跃天数（用于洞察生成）
        cursor.execute(f"""
            SELECT COUNT(DISTINCT strftime('%Y-%m-%d', datetime(view_at + 28800, 'unixepoch'))) as active_days
            FROM {table_name}
        """)
        active_days = cursor.fetchone()[0] or 0

        # 每周观看分布（0=周日，1-6=周一至周六）
        weekday_mapping = {'0': '周日', '1': '周一', '2': '周二', '3': '周三',
                          '4': '周四', '5': '周五', '6': '周六'}
        # 初始化所有星期的默认值为0
        weekly_stats = {day: 0 for day in weekday_mapping.values()}
        cursor.execute(f"""
            SELECT
                strftime('%w', datetime(view_at + 28800, 'unixepoch')) as weekday,
                COUNT(*) as view_count
            FROM {table_name}
            GROUP BY weekday
            ORDER BY weekday
        """)
        # 更新有数据的星期的值
        for row in cursor.fetchall():
            weekly_stats[weekday_mapping[row[0]]] = row[1]

        # 季节性观看模式分析
        cursor.execute(f"""
            SELECT
                CASE
                    WHEN CAST(strftime('%m', datetime(view_at + 28800, 'unixepoch')) AS INTEGER) IN (1,2,3) THEN '春季'
                    WHEN CAST(strftime('%m', datetime(view_at + 28800, 'unixepoch')) AS INTEGER) IN (4,5,6) THEN '夏季'
                    WHEN CAST(strftime('%m', datetime(view_at + 28800, 'unixepoch')) AS INTEGER) IN (7,8,9) THEN '秋季'
                    WHEN CAST(strftime('%m', datetime(view_at + 28800, 'unixepoch')) AS INTEGER) IN (10,11,12) THEN '冬季'
                END as season,
                COUNT(*) as view_count,
                AVG(CASE WHEN progress = -1 THEN duration ELSE progress END) as avg_duration
            FROM {table_name}
            WHERE CAST(strftime('%m', datetime(view_at + 28800, 'unixepoch')) AS INTEGER) BETWEEN 1 AND 12
            GROUP BY season
        """)
        seasonal_patterns = {row[0]: {'view_count': row[1], 'avg_duration': row[2]} for row in cursor.fetchall()}

        # 生成周度统计洞察
        weekly_insights = {}
        if weekly_stats and active_days > 0:
            max_weekday = max(weekly_stats.items(), key=lambda x: x[1])
            min_weekday = min(weekly_stats.items(), key=lambda x: x[1])

            # 计算工作日和周末的平均值
            workday_avg = sum(weekly_stats[day] for day in ['周一', '周二', '周三', '周四', '周五']) / 5
            weekend_avg = sum(weekly_stats[day] for day in ['周六', '周日']) / 2

            if weekend_avg > workday_avg * 1.5:
                week_pattern = "你是一位周末党，倾向于在周末集中补番或观看视频。"
            elif workday_avg > weekend_avg:
                week_pattern = "工作日反而是你观看视频的主要时间，也许是通过B站来缓解工作压力？"
            else:
                week_pattern = "你的观看时间分布很均衡，不管是工作日还是周末都保持着适度的观看习惯。"

            weekly_insights['weekly_pattern'] = f"{week_pattern}其中{max_weekday[0]}是你最喜欢刷B站的日子，平均会看{round(max_weekday[1]/active_days*7, 1)}个视频；而{min_weekday[0]}的观看量最少。"

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "weekly_stats": weekly_stats,
                "seasonal_patterns": seasonal_patterns,
                "insights": weekly_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的周度统计分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'weekly_stats', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/time-slots", summary="获取时段观看分析")
async def get_time_slots(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取时段观看分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含时段观看分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'time_slots')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的时段分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的时段观看数据")

        # 每日时段分布（按小时统计）
        cursor.execute(f"""
            SELECT
                strftime('%H', datetime(view_at + 28800, 'unixepoch')) as hour,
                COUNT(*) as view_count
            FROM {table_name}
            GROUP BY hour
            ORDER BY hour
        """)
        daily_time_slots = {f"{int(row[0])}时": row[1] for row in cursor.fetchall()}

        # 最活跃时段TOP5
        cursor.execute(f"""
            SELECT
                strftime('%H', datetime(view_at + 28800, 'unixepoch')) as hour,
                COUNT(*) as view_count
            FROM {table_name}
            GROUP BY hour
            ORDER BY view_count DESC
            LIMIT 5
        """)
        peak_hours = [{
            "hour": f"{int(row[0])}时",
            "view_count": row[1]
        } for row in cursor.fetchall()]

        # 时间投入分析 - 使用已有的函数
        time_investment = analyze_time_investment(cursor, table_name)

        # 单日最大观看记录
        cursor.execute(f"""
            SELECT
                date(datetime(view_at + 28800, 'unixepoch')) as view_date,
                COUNT(*) as video_count
            FROM {table_name}
            GROUP BY view_date
            ORDER BY video_count DESC
            LIMIT 1
        """)
        max_daily_record = cursor.fetchone()
        max_daily_record = {
            'date': max_daily_record[0],
            'video_count': max_daily_record[1]
        }

        # 生成时段分析洞察
        time_slot_insights = {}
        if daily_time_slots and peak_hours:
            # 将一天分为几个时间段
            morning = sum(daily_time_slots.get(f"{i}时", 0) for i in range(5, 12))
            afternoon = sum(daily_time_slots.get(f"{i}时", 0) for i in range(12, 18))
            evening = sum(daily_time_slots.get(f"{i}时", 0) for i in range(18, 23))
            night = sum(daily_time_slots.get(f"{i}时", 0) for i in range(23, 24)) + sum(daily_time_slots.get(f"{i}时", 0) for i in range(0, 5))

            time_slots = [
                ("清晨和上午", morning),
                ("下午", afternoon),
                ("傍晚和晚上", evening),
                ("深夜", night)
            ]
            primary_slot = max(time_slots, key=lambda x: x[1])

            if primary_slot[0] == "深夜":
                time_advice = "熬夜看视频可能会影响健康，建议调整作息哦！"
            else:
                time_advice = "这个时间段的观看习惯很好，既不影响作息，也能享受视频带来的乐趣。"

            # 获取最高峰时段
            top_hour = peak_hours[0]["hour"] if peak_hours else "未知"

            time_slot_insights['time_preference'] = f"你最喜欢在{primary_slot[0]}观看B站视频，特别是{top_hour}达到观看高峰。{time_advice}"

        # 生成单日观看记录洞察
        if max_daily_record:
            time_slot_insights['daily_record'] = f"在{max_daily_record['date']}这一天，你创下了单日观看{max_daily_record['video_count']}个视频的记录！这可能是一个特别的日子，也许是在追番、学习或者在家放松的一天。"

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "daily_time_slots": daily_time_slots,
                "peak_hours": peak_hours,
                "time_investment": time_investment,
                "max_daily_record": max_daily_record,
                "insights": time_slot_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的时段分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'time_slots', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/continuity", summary="获取观看连续性分析")
async def get_viewing_continuity(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取观看连续性分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含观看连续性分析的数据
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'viewing_continuity')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的观看连续性分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的观看连续性数据")

        # 分析观看连续性
        viewing_continuity = analyze_viewing_continuity(cursor, table_name)

        # 生成连续性洞察
        continuity_insights = generate_continuity_insights(viewing_continuity)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "viewing_continuity": viewing_continuity,
                "insights": continuity_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的观看连续性分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'viewing_continuity', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

def analyze_viewing_details(cursor, table_name: str) -> dict:
    """分析更详细的观看行为，包括设备、总观看时长等
    
    Args:
        cursor: 数据库游标
        table_name: 表名
    
    Returns:
        dict: 详细观看行为分析结果
    """
    # 1. 计算总观看时长（根据progress字段）
    cursor.execute(f"""
        SELECT SUM(CASE WHEN progress = -1 THEN duration ELSE progress END) as total_watch_seconds
        FROM {table_name}
        WHERE progress IS NOT NULL
    """)
    total_seconds = cursor.fetchone()[0] or 0
    total_hours = round(total_seconds / 3600, 1)
    
    # 2. 计算观看B站的总天数
    cursor.execute(f"""
        SELECT COUNT(DISTINCT strftime('%Y-%m-%d', datetime(view_at + 28800, 'unixepoch'))) as total_days
        FROM {table_name}
    """)
    total_days = cursor.fetchone()[0] or 0
    
    # 3. 分析分区观看数据前10
    cursor.execute(f"""
        SELECT 
            main_category, 
            COUNT(*) as view_count,
            SUM(CASE WHEN progress = -1 THEN duration ELSE progress END) as total_progress
        FROM {table_name}
        WHERE main_category IS NOT NULL AND main_category != ''
        GROUP BY main_category
        ORDER BY view_count DESC
        LIMIT 10
    """)
    category_stats = [
        {
            "category": row[0],
            "view_count": row[1],
            "watch_hours": round((row[2] or 0) / 3600, 1)
        } for row in cursor.fetchall()
    ]
    
    # 4. 年度挚爱UP主
    cursor.execute(f"""
        SELECT 
            author_mid, 
            author_name,
            COUNT(*) as view_count,
            SUM(CASE WHEN progress = -1 THEN duration ELSE progress END) as total_progress
        FROM {table_name}
        WHERE author_mid IS NOT NULL
        GROUP BY author_mid
        ORDER BY view_count DESC
        LIMIT 10
    """)
    favorite_up_stats = [
        {
            "mid": row[0],
            "name": row[1],
            "view_count": row[2],
            "watch_hours": round((row[3] or 0) / 3600, 1)
        } for row in cursor.fetchall()
    ]
    
    # 5. 寻找深夜观看记录
    # 第一步：创建临时表存储深夜观看记录
    cursor.execute(f"""
        CREATE TEMPORARY TABLE IF NOT EXISTS temp_night_views AS
        SELECT 
            view_at,
            author_name,
            title,
            strftime('%H', datetime(view_at + 28800, 'unixepoch')) as hour,
            strftime('%M', datetime(view_at + 28800, 'unixepoch')) as minute,
            -- 将凌晨时间(00:00-05:00)的日期调整为前一天
            CASE 
                WHEN strftime('%H', datetime(view_at + 28800, 'unixepoch')) < '05' THEN 
                    date(datetime(view_at + 28800 - 86400, 'unixepoch'))
                ELSE 
                    date(datetime(view_at + 28800, 'unixepoch'))
            END as adjusted_date,
            -- 计算小时+分钟的浮点数时间
            CASE 
                WHEN strftime('%H', datetime(view_at + 28800, 'unixepoch')) < '05' THEN 
                    CAST(strftime('%H', datetime(view_at + 28800, 'unixepoch')) AS REAL) + 24 
                ELSE 
                    CAST(strftime('%H', datetime(view_at + 28800, 'unixepoch')) AS REAL)
            END + CAST(strftime('%M', datetime(view_at + 28800, 'unixepoch')) AS REAL)/100.0 as hour_with_minute
        FROM {table_name}
        WHERE 
            strftime('%H', datetime(view_at + 28800, 'unixepoch')) >= '23' OR 
            strftime('%H', datetime(view_at + 28800, 'unixepoch')) < '05'
    """)
    
    # 第二步：创建临时表存储每天最晚的观看时间
    cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS temp_latest_per_day AS
        SELECT 
            adjusted_date,
            MAX(hour_with_minute) as latest_hour_with_minute
        FROM temp_night_views
        GROUP BY adjusted_date
    """)
    
    # 第三步：查询每天最晚的观看记录
    cursor.execute("""
        SELECT 
            t.adjusted_date as date,
            strftime('%H:%M', datetime(t.view_at + 28800, 'unixepoch')) as time,
            t.author_name,
            t.title,
            t.hour,
            t.minute,
            t.hour_with_minute
        FROM temp_night_views t
        JOIN temp_latest_per_day l ON 
            t.adjusted_date = l.adjusted_date AND 
            t.hour_with_minute = l.latest_hour_with_minute
        ORDER BY t.hour_with_minute DESC
        LIMIT 10
    """)
    
    late_night_views = [
        {
            "date": row[0],
            "time": row[1],
            "author": row[2],
            "title": row[3],
            "hour": int(row[4]),
            "minute": row[5],
            "hour_with_minute": float(row[6])
        } for row in cursor.fetchall()
    ]
    
    # 清理临时表
    cursor.execute("DROP TABLE IF EXISTS temp_night_views")
    cursor.execute("DROP TABLE IF EXISTS temp_latest_per_day")
    
    # 6. 各时间段的活跃天数百分比
    cursor.execute(f"""
        SELECT 
            CASE 
                WHEN strftime('%H', datetime(view_at + 28800, 'unixepoch')) BETWEEN '05' AND '11' THEN '上午'
                WHEN strftime('%H', datetime(view_at + 28800, 'unixepoch')) BETWEEN '12' AND '17' THEN '下午'
                WHEN strftime('%H', datetime(view_at + 28800, 'unixepoch')) BETWEEN '18' AND '22' THEN '晚上'
                ELSE '深夜'
            END as time_slot,
            COUNT(DISTINCT strftime('%Y-%m-%d', datetime(view_at + 28800, 'unixepoch'))) as active_days
        FROM {table_name}
        GROUP BY time_slot
    """)
    time_slot_days = {}
    for row in cursor.fetchall():
        time_slot_days[row[0]] = {
            "days": row[1],
            "percentage": round(row[1] / total_days * 100, 1) if total_days > 0 else 0
        }
    
    # 7. 查询最常用的设备信息（如果有）
    cursor.execute(f"""
        SELECT 
            CASE 
                WHEN dt IN (1, 3, 5, 7) THEN '手机'
                WHEN dt = 2 THEN '网页'
                WHEN dt IN (4, 6) THEN '平板'
                WHEN dt = 33 THEN '电视'
                ELSE '其他'
            END as platform,
            COUNT(*) as count
        FROM {table_name}
        GROUP BY platform
        ORDER BY count DESC
        LIMIT 3
    """)
    devices = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
        
    return {
        "total_watch_hours": total_hours,
        "total_days": total_days,
        "top_categories": category_stats,
        "favorite_up_users": favorite_up_stats,
        "late_night_views": late_night_views,
        "time_slot_activity": time_slot_days,
        "devices": devices
    }

def generate_viewing_report(viewing_details: dict) -> dict:
    """根据观看数据生成综合报告和亮点
    
    Args:
        viewing_details: 观看详细数据
    
    Returns:
        dict: 包含各种总结语句的报告
    """
    report = {}
    
    # 1. 总观看时长和天数总结
    report["total_summary"] = f"和B站共度的{viewing_details['total_days']}天里，你观看超过{viewing_details['total_watch_hours']}小时的内容"
    
    # 2. 时间段访问情况
    time_slots = viewing_details["time_slot_activity"]
    max_slot = max(time_slots.items(), key=lambda x: x[1]["percentage"]) if time_slots else None
    
    if max_slot:
        report["time_slot_summary"] = f"{max_slot[0]}时段访问B站天数超过{max_slot[1]['percentage']}%"
    
    # 3. 深夜观看记录
    if viewing_details["late_night_views"]:
        # late_night_views已按照最晚时间排序，首条记录是最晚的
        latest_view = viewing_details["late_night_views"][0]
        report["late_night_summary"] = f"{latest_view['date']}你迟迟不肯入睡，{latest_view['time']}还在看{latest_view['author']}的《{latest_view['title']}》"
    
    # 4. 分区观看总结
    if viewing_details["top_categories"]:
        top_category = viewing_details["top_categories"][0]
        report["category_summary"] = f"你最喜欢的分区是{top_category['category']}，共观看{top_category['view_count']}个视频，时长{top_category['watch_hours']}小时"
    
    # 5. 年度挚爱UP总结
    if viewing_details["favorite_up_users"]:
        top_up = viewing_details["favorite_up_users"][0]
        report["up_summary"] = f"年度挚爱UP主是{top_up['name']}，共观看{top_up['view_count']}个视频，时长{top_up['watch_hours']}小时"
    
    # 6. 设备使用总结
    if viewing_details["devices"]:
        top_device = viewing_details["devices"][0]
        report["device_summary"] = f"你最常用的观看设备是{top_device['name']}，共使用{top_device['count']}次"
    
    return report

@router.get("/viewing/", summary="获取观看行为数据分析")
async def get_viewing_details(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取观看行为数据分析
    
    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据
    
    Returns:
        dict: 包含观看行为分析的详细数据和总结报告
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # 如果启用缓存，尝试从缓存获取完整响应
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'viewing_details')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的观看行为分析数据")
                return cached_response
        
        print(f"开始分析 {target_year} 年的观看行为数据")
        
        # 获取详细观看数据
        viewing_details = analyze_viewing_details(cursor, table_name)
        
        # 生成综合报告
        viewing_report = generate_viewing_report(viewing_details)
        
        # 构建完整响应
        response = {
            "status": "success",
            "data": {
                "details": viewing_details,
                "report": viewing_report,
                "year": target_year,
                "available_years": available_years
            }
        }
        
        # 无论是否启用缓存，都更新缓存数据
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的观看行为分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'viewing_details', response)
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/watch-counts", summary="获取重复观看分析")
async def get_viewing_watch_counts(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取用户重复观看分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含重复观看分析结果的响应
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'watch_counts')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的重复观看分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的重复观看数据")

        # 获取重复观看分析数据
        watch_count_data = analyze_video_watch_counts(cursor, table_name)

        # 生成重复观看洞察
        watch_count_insights = generate_watch_count_insights(watch_count_data)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "watch_counts": watch_count_data,
                "insights": watch_count_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的重复观看分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'watch_counts', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/completion-rates", summary="获取视频完成率分析")
async def get_viewing_completion_rates(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取用户视频完成率分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含视频完成率分析结果的响应
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'completion_rates')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的视频完成率分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的视频完成率数据")

        # 获取视频完成率分析数据
        completion_data = analyze_completion_rates(cursor, table_name)

        # 生成完成率洞察
        completion_insights = generate_completion_insights(completion_data)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "completion_rates": completion_data,
                "insights": completion_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的视频完成率分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'completion_rates', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

def analyze_author_completion_rates(cursor, table_name: str) -> dict:
    """专门分析UP主完成率数据，使用智能综合评分算法"""
    # 获取表结构
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1]: idx for idx, col in enumerate(cursor.fetchall())}

    # 确保必要的列存在
    required_columns = ['duration', 'progress', 'author_name', 'author_mid']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Required column '{col}' not found in table {table_name}")

    # 获取所有历史记录
    cursor.execute(f"SELECT * FROM {table_name}")
    histories = cursor.fetchall()

    author_stats = {}

    for history in histories:
        # 获取并转换数据类型
        try:
            duration = float(history[columns['duration']]) if history[columns['duration']] else 0
            progress = float(history[columns['progress']]) if history[columns['progress']] else 0
            author_name = history[columns['author_name']]
            author_mid = history[columns['author_mid']]
        except (ValueError, TypeError):
            continue

        # 计算完成率
        if progress == -1:
            completion_rate = 100
        else:
            completion_rate = (progress / duration * 100) if duration > 0 else 0

        # UP主统计
        if author_name and author_mid:
            if author_name not in author_stats:
                author_stats[author_name] = {
                    "author_mid": author_mid,
                    "video_count": 0,
                    "total_completion": 0,
                    "fully_watched": 0
                }
            stats = author_stats[author_name]
            stats["video_count"] += 1
            stats["total_completion"] += completion_rate
            if completion_rate >= 90:
                stats["fully_watched"] += 1

    # 计算UP主平均完成率和完整观看率，并按观看数量筛选
    filtered_authors = {}
    for name, stats in author_stats.items():
        if stats["video_count"] >= 5:  # 只保留观看数量>=5的UP主
            stats["average_completion_rate"] = round(stats["total_completion"] / stats["video_count"], 2)
            stats["fully_watched_rate"] = round(stats["fully_watched"] / stats["video_count"] * 100, 2)
            filtered_authors[name] = stats

    # 使用新的智能评分算法
    scored_authors = calculate_comprehensive_author_scores(filtered_authors)

    return {
        "most_watched_authors": scored_authors["most_watched_authors"],
        "highest_completion_authors": scored_authors["highest_completion_authors"],
        "most_valuable_authors": scored_authors["most_valuable_authors"],
        "potential_authors": scored_authors["potential_authors"]
    }

def calculate_comprehensive_author_scores(authors_data: dict) -> dict:
    """计算UP主综合评分并分类"""
    import math
    
    if not authors_data:
        return {
            "most_watched_authors": {},
            "highest_completion_authors": {},
            "most_valuable_authors": {},
            "potential_authors": {}
        }
    
    # 计算观看数量的最大值和最小值，用于标准化
    view_counts = [stats["video_count"] for stats in authors_data.values()]
    max_views = max(view_counts)
    min_views = min(view_counts)
    view_range = max_views - min_views if max_views > min_views else 1
    
    # 为每个UP主计算综合评分
    scored_authors = {}
    for name, stats in authors_data.items():
        # 1. 观看数量标准化 (0-100)
        normalized_views = (stats["video_count"] - min_views) / view_range * 100
        
        # 2. 置信度计算 (避免小样本偏差)
        confidence = min(1.0, stats["video_count"] / 20)
        
        # 3. 综合评分计算
        # 权重: 观看数量25%, 完成率50%, 完整观看率25%
        comprehensive_score = (
            normalized_views * 0.25 +
            stats["average_completion_rate"] * 0.5 +
            stats["fully_watched_rate"] * 0.25
        ) * confidence
        
        # 4. 计算特殊指标
        # 忠诚度指标：观看数量 * 完成率的平方根
        loyalty_score = stats["video_count"] * math.sqrt(stats["average_completion_rate"] / 100)
        
        # 质量指标：完成率和完整观看率的加权平均
        quality_score = stats["average_completion_rate"] * 0.7 + stats["fully_watched_rate"] * 0.3
        
        scored_authors[name] = {
            **stats,
            "comprehensive_score": round(comprehensive_score, 2),
            "loyalty_score": round(loyalty_score, 2),
            "quality_score": round(quality_score, 2),
            "confidence": round(confidence, 2),
            "normalized_views": round(normalized_views, 2)
        }
    
    # 按不同维度排序分类
    # 1. 观看次数最多的UP主 (传统排序)
    most_watched_authors = dict(sorted(
        scored_authors.items(),
        key=lambda x: x[1]["video_count"],
        reverse=True
    )[:10])
    
    # 2. 完成率最高的UP主 (传统排序)
    highest_completion_authors = dict(sorted(
        scored_authors.items(),
        key=lambda x: x[1]["average_completion_rate"],
        reverse=True
    )[:10])
    
    # 3. 最有价值UP主 (综合评分最高)
    most_valuable_authors = dict(sorted(
        scored_authors.items(),
        key=lambda x: x[1]["comprehensive_score"],
        reverse=True
    )[:10])
    
    # 4. 潜力UP主 (高质量但观看数量不是最多的)
    potential_authors = {}
    for name, stats in scored_authors.items():
        # 筛选条件：质量评分>85，但观看数量不在前3
        if (stats["quality_score"] > 85 and
            stats["video_count"] < sorted(view_counts, reverse=True)[min(2, len(view_counts)-1)]):
            potential_authors[name] = stats
    
    # 潜力UP主按质量评分排序，取前5
    potential_authors = dict(sorted(
        potential_authors.items(),
        key=lambda x: x[1]["quality_score"],
        reverse=True
    )[:5])
    
    return {
        "most_watched_authors": most_watched_authors,
        "highest_completion_authors": highest_completion_authors,
        "most_valuable_authors": most_valuable_authors,
        "potential_authors": potential_authors
    }

def generate_author_completion_insights(author_data: dict) -> dict:
    """生成UP主完成率相关的智能洞察"""
    insights = {}

    try:
        # 1. 最有价值UP主洞察（综合评分最高）
        most_valuable = author_data.get("most_valuable_authors", {})
        if most_valuable:
            top_valuable = next(iter(most_valuable.items()), None)
            if top_valuable:
                name, stats = top_valuable
                insights["most_valuable_author"] = (
                    f"综合来看，{name}是你最有价值的UP主，观看了{stats.get('video_count', 0)}个视频，"
                    f"平均完成率高达{stats.get('average_completion_rate', 0)}%，"
                    f"完整观看率{stats.get('fully_watched_rate', 0)}%，综合评分{stats.get('comprehensive_score', 0)}分。"
                )

        # 2. 观看最多UP主洞察（如果与最有价值UP主不同）
        most_watched = author_data.get("most_watched_authors", {})
        if most_watched:
            top_watched = next(iter(most_watched.items()), None)
            if top_watched and (not most_valuable or top_watched[0] != next(iter(most_valuable.keys()))):
                name, stats = top_watched
                insights["most_watched_author"] = (
                    f"你观看最多的UP主是{name}，观看了{stats.get('video_count', 0)}个视频，"
                    f"平均完成率为{stats.get('average_completion_rate', 0)}%，是你的喜爱观看对象。"
                )

        # 3. 生成综合观看行为分析
        if most_valuable:
            total_authors = len(author_data.get("most_watched_authors", {}))
            insights["viewing_behavior_summary"] = (
                f"在你经常观看的{total_authors}位UP主中，你展现出了很好的内容鉴赏能力，"
                f"特别偏爱高质量内容，对喜欢的UP主有很高的完成度和喜爱度。"
            )

    except Exception as e:
        print(f"Error generating author completion insights: {str(e)}")
        insights["basic_author"] = "暂时无法生成详细的UP主完成率分析。"

    return insights

@router.get("/author-completion", summary="获取UP主完成率分析")
async def get_viewing_author_completion(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取UP主完成率分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含UP主完成率分析结果的响应
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'author_completion')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的UP主完成率分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的UP主完成率数据")

        # 获取UP主完成率分析数据
        author_data = analyze_author_completion_rates(cursor, table_name)

        # 生成UP主完成率洞察
        author_insights = generate_author_completion_insights(author_data)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "completion_rates": author_data,
                "insights": author_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的UP主完成率分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'author_completion', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

def analyze_tag_analysis(cursor, table_name: str) -> dict:
    """专门分析标签数据，包括分布和完成率"""
    # 获取表结构
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1]: idx for idx, col in enumerate(cursor.fetchall())}

    # 确保必要的列存在
    required_columns = ['duration', 'progress', 'tag_name']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Required column '{col}' not found in table {table_name}")

    # 获取所有历史记录
    cursor.execute(f"SELECT * FROM {table_name}")
    histories = cursor.fetchall()

    tag_stats = {}
    tag_distribution = {}

    for history in histories:
        # 获取并转换数据类型
        try:
            duration = float(history[columns['duration']]) if history[columns['duration']] else 0
            progress = float(history[columns['progress']]) if history[columns['progress']] else 0
            tag_name = history[columns['tag_name']]
        except (ValueError, TypeError):
            continue

        # 计算完成率
        if progress == -1:
            completion_rate = 100
        else:
            completion_rate = (progress / duration * 100) if duration > 0 else 0

        # 标签分布统计
        if tag_name:
            tag_distribution[tag_name] = tag_distribution.get(tag_name, 0) + 1

            # 标签完成率统计
            if tag_name not in tag_stats:
                tag_stats[tag_name] = {
                    "video_count": 0,
                    "total_completion": 0,
                    "fully_watched": 0
                }
            stats = tag_stats[tag_name]
            stats["video_count"] += 1
            stats["total_completion"] += completion_rate
            if completion_rate >= 90:
                stats["fully_watched"] += 1

    # 计算标签平均完成率和完整观看率，并按观看数量筛选
    filtered_tags = {}
    for tag, stats in tag_stats.items():
        if stats["video_count"] >= 5:  # 只保留视频数量>=5的标签
            stats["average_completion_rate"] = round(stats["total_completion"] / stats["video_count"], 2)
            stats["fully_watched_rate"] = round(stats["fully_watched"] / stats["video_count"] * 100, 2)
            filtered_tags[tag] = stats

    # 获取完成率最高的标签
    top_completion_tags = dict(sorted(
        filtered_tags.items(),
        key=lambda x: x[1]["average_completion_rate"],
        reverse=True
    )[:10])

    # 获取观看最多的标签
    top_watched_tags = dict(sorted(
        tag_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10])

    return {
        "tag_distribution": top_watched_tags,
        "tag_completion_rates": top_completion_tags
    }

def generate_tag_analysis_insights(tag_data: dict) -> dict:
    """生成标签分析相关的洞察"""
    insights = {}

    try:
        # 标签偏好洞察
        tag_distribution = tag_data.get("tag_distribution", {})
        if tag_distribution:
            top_tag = next(iter(tag_distribution.items()), None)
            if top_tag:
                insights["tag_preference"] = (
                    f"你最喜欢观看{top_tag[0]}分区的视频，共观看了{top_tag[1]}个视频。"
                )

        # 标签完成率洞察
        tag_completion = tag_data.get("tag_completion_rates", {})
        if tag_completion:
            top_completion = next(iter(tag_completion.items()), None)
            if top_completion:
                insights["tag_completion"] = (
                    f"在经常观看的分区中，你对{top_completion[0]}分区的视频最感兴趣，"
                    f"平均完成率达到{top_completion[1].get('average_completion_rate', 0)}%，"
                    f"观看过{top_completion[1].get('video_count', 0)}个该分区的视频。"
                )

    except Exception as e:
        print(f"Error generating tag analysis insights: {str(e)}")
        insights["basic_tag"] = "暂时无法生成详细的标签分析。"

    return insights

@router.get("/tag-analysis", summary="获取标签分析")
async def get_viewing_tag_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取标签分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含标签分析结果的响应
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'tag_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的标签分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的标签数据")

        # 获取标签分析数据
        tag_data = analyze_tag_analysis(cursor, table_name)

        # 生成标签分析洞察
        tag_insights = generate_tag_analysis_insights(tag_data)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "watch_counts": {"tag_distribution": tag_data["tag_distribution"]},
                "completion_rates": {"tag_completion_rates": tag_data["tag_completion_rates"]},
                "insights": tag_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的标签分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'tag_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

def analyze_duration_analysis(cursor, table_name: str) -> dict:
    """专门分析视频时长数据"""
    # 获取表结构
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1]: idx for idx, col in enumerate(cursor.fetchall())}

    # 确保必要的列存在
    required_columns = ['duration', 'view_at']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Required column '{col}' not found in table {table_name}")

    # 获取所有历史记录
    cursor.execute(f"SELECT * FROM {table_name}")
    histories = cursor.fetchall()

    print(f"Debug: 总共获取到 {len(histories)} 条历史记录")
    if len(histories) > 0:
        print(f"Debug: 第一条记录示例: {histories[0]}")
        print(f"Debug: 列映射: {columns}")

    # 时段分类
    time_periods = {
        '凌晨': {'start': 0, 'end': 6},
        '上午': {'start': 6, 'end': 12},
        '下午': {'start': 12, 'end': 18},
        '晚上': {'start': 18, 'end': 24}
    }

    # 初始化时长相关性数据
    duration_correlation = {}
    for period in time_periods.keys():
        duration_correlation[period] = {
            '短视频': {'video_count': 0, 'total_duration': 0, 'avg_duration': 0},
            '中等视频': {'video_count': 0, 'total_duration': 0, 'avg_duration': 0},
            '长视频': {'video_count': 0, 'total_duration': 0, 'avg_duration': 0}
        }

    processed_count = 0
    valid_count = 0

    for history in histories:
        processed_count += 1
        try:
            duration = float(history[columns['duration']]) if history[columns['duration']] else 0
            view_at = history[columns['view_at']]

            if not view_at or duration <= 0:
                if processed_count <= 5:  # 只打印前5条的调试信息
                    print(f"Debug: 跳过记录 {processed_count}: view_at={view_at}, duration={duration}")
                continue

            # 解析观看时间 - view_at 是 Unix 时间戳
            from datetime import datetime
            try:
                # view_at 是 Unix 时间戳，需要转换为 datetime 对象
                view_time = datetime.fromtimestamp(int(view_at))
                hour = view_time.hour
            except (ValueError, TypeError, OSError):
                continue

            # 确定时段
            period = None
            for p, time_range in time_periods.items():
                if time_range['start'] <= hour < time_range['end']:
                    period = p
                    break

            if not period:
                continue

            # 确定时长类型
            if duration < 300:  # 5分钟以下
                duration_type = '短视频'
            elif duration < 1200:  # 20分钟以下
                duration_type = '中等视频'
            else:
                duration_type = '长视频'

            # 更新统计
            stats = duration_correlation[period][duration_type]
            stats['video_count'] += 1
            stats['total_duration'] += duration
            valid_count += 1

            if valid_count <= 5:  # 只打印前5条有效记录的调试信息
                print(f"Debug: 有效记录 {valid_count}: period={period}, duration_type={duration_type}, duration={duration}")

        except (ValueError, TypeError):
            if processed_count <= 5:
                print(f"Debug: 数据转换失败，记录 {processed_count}")
            continue

    # 计算平均时长
    for period in duration_correlation:
        for duration_type in duration_correlation[period]:
            stats = duration_correlation[period][duration_type]
            if stats['video_count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['video_count']

    print(f"Debug: 处理完成 - 总记录数: {processed_count}, 有效记录数: {valid_count}")
    print(f"Debug: 最终统计结果: {duration_correlation}")

    return duration_correlation

def generate_duration_analysis_insights(duration_data: dict) -> dict:
    """生成视频时长分析相关的洞察"""
    insights = {}

    try:
        # 计算总体时长偏好
        total_counts = {'短视频': 0, '中等视频': 0, '长视频': 0}

        for period in duration_data:
            for duration_type in duration_data[period]:
                total_counts[duration_type] += duration_data[period][duration_type]['video_count']

        if sum(total_counts.values()) > 0:
            # 找出最喜欢的时长类型
            preferred_type = max(total_counts.items(), key=lambda x: x[1])
            total_videos = sum(total_counts.values())
            preference_rate = round(preferred_type[1] / total_videos * 100, 1)

            # 找出最活跃的时段和对应的时长偏好
            max_period_count = 0
            max_period = None
            max_period_type = None

            for period in duration_data:
                for duration_type in duration_data[period]:
                    count = duration_data[period][duration_type]['video_count']
                    if count > max_period_count:
                        max_period_count = count
                        max_period = period
                        max_period_type = duration_type

            insights["duration_preference"] = (
                f"你最喜欢观看{preferred_type[0]}，占总观看量的{preference_rate}%。"
                f"特别是在{max_period}时段，你更偏向于观看{max_period_type}。"
            )

    except Exception as e:
        print(f"Error generating duration analysis insights: {str(e)}")
        insights["basic_duration"] = "暂时无法生成详细的时长分析。"

    return insights

@router.get("/duration-analysis", summary="获取视频时长分析")
async def get_viewing_duration_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取视频时长分析

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含视频时长分析结果的响应
    """
    # 验证年份并获取表名
    table_name, target_year, available_years = validate_year_and_get_table(year)
    if table_name is None:
        return available_years  # 这里是错误响应

    conn = get_db()
    try:
        cursor = conn.cursor()

        # 如果启用缓存，尝试从缓存获取
        if use_cache:
            from .title_pattern_discovery import pattern_cache
            cached_response = pattern_cache.get_cached_patterns(table_name, 'duration_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的视频时长分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的视频时长数据")

        # 获取视频时长分析数据
        duration_data = analyze_duration_analysis(cursor, table_name)

        # 生成时长分析洞察
        duration_insights = generate_duration_analysis_insights(duration_data)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "duration_correlation": duration_data,
                "insights": duration_insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的视频时长分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'duration_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
