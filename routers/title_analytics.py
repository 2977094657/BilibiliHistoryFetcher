import sqlite3
from collections import Counter
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import jieba
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from snownlp import SnowNLP

from scripts.utils import load_config, get_output_path
from .title_pattern_discovery import discover_interaction_patterns

router = APIRouter()
config = load_config()

def get_db():
    """获取数据库连接"""
    db_path = get_output_path(config['db_file'])
    return sqlite3.connect(db_path)

def analyze_keywords(titles_data: List[tuple]) -> List[Tuple[str, int]]:
    """
    从标题数据中提取关键词及其频率
    
    Args:
        titles_data: 包含(title, duration, progress, tag_name, view_at)的元组列表
    
    Returns:
        List[Tuple[str, int]]: 关键词和频率的列表
    """
    # 停用词列表（可以根据需要扩展）
    stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    
    # 所有标题分词后的结果
    all_words = []
    for title_data in titles_data:
        title = title_data[0]  # 标题在元组的第一个位置
        if not title:  # 跳过空标题
            continue
        words = jieba.cut(title)
        # 过滤停用词和单字词（通常单字词不能很好地表达含义）
        words = [w for w in words if w not in stop_words and len(w) > 1]
        all_words.extend(words)
    
    # 统计词频
    word_freq = Counter(all_words)
    
    # 返回前20个最常见的词
    return word_freq.most_common(20)

def analyze_completion_rates(titles_data: List[tuple]) -> Dict:
    """
    分析标题特征与完成率的关系
    
    Args:
        titles_data: 包含(title, duration, progress, tag_name, view_at)的元组列表
    
    Returns:
        Dict: 完成率分析结果
    """
    # 计算每个视频的完成率
    completion_rates = []
    titles = []
    for title_data in titles_data:
        title = title_data[0]
        duration = float(title_data[1]) if title_data[1] is not None else 0
        progress = float(title_data[2]) if title_data[2] is not None else 0

        if duration and duration > 0:  # 确保duration有效且大于0
            if progress == -1.0:  # progress为-1表示看完了
                completion_rate = 1.0
            elif progress > 0:
                completion_rate = min(progress / duration, 1.0)  # 限制最大值为1
            else:
                completion_rate = 0.0
            completion_rates.append(completion_rate)
            titles.append(title)
    
    # 提取关键词
    keywords = analyze_keywords([(title,) for title in titles])
    
    # 分析包含每个关键词的视频的平均完成率
    keyword_completion_rates = {}
    for keyword, _ in keywords:
        rates = []
        for title, rate in zip(titles, completion_rates):
            if keyword in title:
                rates.append(rate)
        if rates:  # 如果有包含该关键词的视频
            avg_rate = sum(rates) / len(rates)
            keyword_completion_rates[keyword] = {
                'average_completion_rate': avg_rate,
                'video_count': len(rates)
            }
    
    return keyword_completion_rates

def generate_insights(keywords: List[Tuple[str, int]], completion_rates: Dict) -> List[str]:
    """
    根据关键词和完成率生成洞察
    
    Args:
        keywords: 关键词和频率的列表
        completion_rates: 完成率分析结果
    
    Returns:
        List[str]: 洞察列表
    """
    insights = []
    
    # 1. 关键词频率洞察
    top_keywords = [(word, count) for word, count in keywords[:5]]
    if top_keywords:
        insights.append(f"在您观看的视频中，最常出现的关键词是：{', '.join([f'{word}({count}次)' for word, count in top_keywords])}")
    
    # 2. 完成率洞察
    if completion_rates:
        # 按完成率排序
        sorted_rates = sorted(
            [(k, v['average_completion_rate'], v['video_count']) 
             for k, v in completion_rates.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 高完成率关键词（前3个）
        high_completion = sorted_rates[:3]
        if high_completion:
            insights.append(f"包含关键词 {', '.join([f'{k}({rate:.1%})' for k, rate, count in high_completion])} 的视频往往会被您看完。")
        
        # 低完成率关键词（后3个）
        low_completion = sorted_rates[-3:]
        low_completion.reverse()  # 从低到高显示
        if low_completion:
            insights.append(f"而包含关键词 {', '.join([f'{k}({rate:.1%})' for k, rate, count in low_completion])} 的视频较少被看完。")
    
    return insights

def analyze_title_length(cursor, table_name: str) -> dict:
    """分析标题长度与观看行为的关系"""
    cursor.execute(f"""
        SELECT title, duration, progress
        FROM {table_name}
        WHERE duration > 0 AND title IS NOT NULL
        AND strftime('%Y', datetime(view_at, 'unixepoch')) = ?
    """, (table_name.split('_')[-1],))
    
    length_stats = defaultdict(lambda: {'count': 0, 'completion_rates': []})
    
    for title, duration, progress in cursor.fetchall():
        length = len(title)
        completion_rate = progress / duration if duration > 0 else 0
        length_group = (length // 5) * 5  # 按5个字符分组
        length_stats[length_group]['count'] += 1
        length_stats[length_group]['completion_rates'].append(completion_rate)
    
    # 计算每个长度组的平均完成率
    results = {}
    for length_group, stats in length_stats.items():
        avg_completion = np.mean(stats['completion_rates'])
        results[f"{length_group}-{length_group+4}字"] = {
            'count': stats['count'],
            'avg_completion_rate': avg_completion
        }
    
    # 找出最佳长度范围
    best_length = max(results.items(), key=lambda x: x[1]['avg_completion_rate'])
    most_common = max(results.items(), key=lambda x: x[1]['count'])
    
    return {
        'length_stats': results,
        'best_length': best_length[0],
        'most_common_length': most_common[0],
        'insights': [
            f"标题长度在{best_length[0]}的视频最容易被你看完，平均完成率为{best_length[1]['avg_completion_rate']:.1%}",
            f"你观看的视频中，标题长度最常见的是{most_common[0]}，共有{most_common[1]['count']}个视频"
        ]
    }

def analyze_title_sentiment(cursor, table_name: str) -> dict:
    """分析标题情感与观看行为的关系"""
    cursor.execute(f"""
        SELECT title, duration, progress
        FROM {table_name}
        WHERE duration > 0 AND title IS NOT NULL
        AND strftime('%Y', datetime(view_at, 'unixepoch')) = ?
    """, (table_name.split('_')[-1],))
    
    sentiment_stats = {
        '积极': {'count': 0, 'completion_rates': []},
        '中性': {'count': 0, 'completion_rates': []},
        '消极': {'count': 0, 'completion_rates': []}
    }
    
    for title, duration, progress in cursor.fetchall():
        sentiment = SnowNLP(title).sentiments
        completion_rate = progress / duration if duration > 0 else 0
        
        # 情感分类
        if sentiment > 0.6:
            category = '积极'
        elif sentiment < 0.4:
            category = '消极'
        else:
            category = '中性'
            
        sentiment_stats[category]['count'] += 1
        sentiment_stats[category]['completion_rates'].append(completion_rate)
    
    # 计算每种情感的平均完成率
    results = {}
    for sentiment, stats in sentiment_stats.items():
        if stats['count'] > 0:
            results[sentiment] = {
                'count': stats['count'],
                'avg_completion_rate': np.mean(stats['completion_rates'])
            }
    
    # 找出最受欢迎的情感类型
    best_sentiment = max(results.items(), key=lambda x: x[1]['avg_completion_rate'])
    most_common = max(results.items(), key=lambda x: x[1]['count'])
    
    return {
        'sentiment_stats': results,
        'best_sentiment': best_sentiment[0],
        'most_common_sentiment': most_common[0],
        'insights': [
            f"{best_sentiment[0]}情感的视频最容易引起你的兴趣，平均完成率为{best_sentiment[1]['avg_completion_rate']:.1%}",
            f"在你观看的视频中，{most_common[0]}情感的内容最多，共有{most_common[1]['count']}个视频"
        ]
    }

def analyze_title_trends(cursor, table_name: str) -> dict:
    """分析标题趋势与观看行为的关系"""
    cursor.execute(f"""
        SELECT title, duration, progress, view_at
        FROM {table_name}
        WHERE duration > 0 AND title IS NOT NULL
        AND strftime('%Y', datetime(view_at, 'unixepoch')) = ?
        ORDER BY view_at ASC
    """, (table_name.split('_')[-1],))
    
    # 按月分组的关键词统计和视频计数
    monthly_keywords = defaultdict(lambda: defaultdict(int))
    monthly_video_count = defaultdict(int)  # 新增：每月视频计数
    
    for title, duration, progress, view_at in cursor.fetchall():
        month = datetime.fromtimestamp(view_at).strftime('%Y-%m')
        monthly_video_count[month] += 1  # 新增：增加月度视频计数
        words = jieba.cut(title)
        for word in words:
            if len(word) > 1:  # 排除单字词
                monthly_keywords[month][word] += 1
    
    # 分析每个月的热门关键词
    trending_keywords = {}
    for month, keywords in monthly_keywords.items():
        # 获取当月TOP5关键词
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        trending_keywords[month] = {
            'top_keywords': top_keywords,
            'total_videos': monthly_video_count[month]  # 修改：使用实际的月度视频数量
        }
    
    # 识别关键词趋势
    all_months = sorted(trending_keywords.keys())
    if len(all_months) >= 2:
        first_month = all_months[0]
        last_month = all_months[-1]
        
        # 计算关键词增长率
        first_keywords = set(k for k, _ in trending_keywords[first_month]['top_keywords'])
        last_keywords = set(k for k, _ in trending_keywords[last_month]['top_keywords'])
        
        new_trending = last_keywords - first_keywords
        fading = first_keywords - last_keywords
        consistent = first_keywords & last_keywords
        
        trend_insights = []
        if new_trending:
            trend_insights.append(f"新兴关键词: {', '.join(new_trending)}")
        if consistent:
            trend_insights.append(f"持续热门: {', '.join(consistent)}")
        if fading:
            trend_insights.append(f"减少关注: {', '.join(fading)}")
    else:
        trend_insights = ["数据量不足以分析趋势"]
    
    return {
        'monthly_trends': trending_keywords,
        'insights': trend_insights
    }

def analyze_title_interaction(cursor, table_name: str) -> dict:
    """分析标题与用户互动的关系"""
    cursor.execute(f"""
        SELECT title, duration, progress, tag_name, view_at
        FROM {table_name}
        WHERE duration > 0 AND title IS NOT NULL
        AND strftime('%Y', datetime(view_at, 'unixepoch')) = ?
    """, (table_name.split('_')[-1],))
    
    titles_data = cursor.fetchall()
    
    # 使用互动模式发现功能，不再传递table_name参数
    discovered_patterns = discover_interaction_patterns(titles_data)
    
    interaction_stats = defaultdict(lambda: {'count': 0, 'completion_rates': []})
    
    for title, duration, progress, *_ in titles_data:
        completion_rate = progress / duration if duration > 0 else 0
        found_pattern = False
        
        for pattern_type, pattern_info in discovered_patterns.items():
            if any(keyword in title for keyword in pattern_info['keywords']):
                interaction_stats[pattern_type]['count'] += 1
                interaction_stats[pattern_type]['completion_rates'].append(completion_rate)
                found_pattern = True
        
        if not found_pattern:
            interaction_stats['其他']['count'] += 1
            interaction_stats['其他']['completion_rates'].append(completion_rate)
    
    results = {}
    for pattern, stats in interaction_stats.items():
        if stats['count'] > 0:
            results[pattern] = {
                'count': stats['count'],
                'avg_completion_rate': np.mean(stats['completion_rates']),
                'keywords': discovered_patterns[pattern]['keywords'] if pattern in discovered_patterns else []
            }
    
    best_pattern = max(results.items(), key=lambda x: x[1]['avg_completion_rate'])
    most_common = max(results.items(), key=lambda x: x[1]['count'])
    
    return {
        'interaction_stats': results,
        'best_pattern': best_pattern[0],
        'most_common_pattern': most_common[0],
        'insights': [
            f"{best_pattern[0]}互动方式的标题最容易引起互动，平均完成率为{best_pattern[1]['avg_completion_rate']:.1%}",
            f"在你观看的视频中，{most_common[0]}互动方式最常见，共有{most_common[1]['count']}个视频"
        ]
    }

def validate_year_and_get_table(year: Optional[int]) -> tuple:
    """验证年份并返回表名和可用年份列表

    Args:
        year: 要验证的年份，None表示使用最新年份

    Returns:
        tuple: (table_name, target_year, available_years) 或 (None, None, error_response)
    """
    # 导入已有的函数，避免重复代码
    from .viewing_analytics import get_available_years

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

@router.get("/keyword-analysis", summary="获取标题关键词分析")
async def get_keyword_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取标题关键词分析数据

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含关键词分析的数据
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
            cached_response = pattern_cache.get_cached_patterns(table_name, 'keyword_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的关键词分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的标题关键词数据")

        # 获取所有标题数据
        cursor.execute(f"""
            SELECT title, duration, progress
            FROM {table_name}
            WHERE title IS NOT NULL AND title != ''
        """)

        titles_data = cursor.fetchall()

        if not titles_data:
            return {
                "status": "error",
                "message": "未找到任何有效的标题数据"
            }

        # 分词和关键词提取
        keywords = analyze_keywords(titles_data)

        # 分析完成率
        completion_analysis = analyze_completion_rates(titles_data)

        # 生成洞察
        insights = generate_insights(keywords, completion_analysis)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "keyword_analysis": {
                    "top_keywords": [{"word": word, "count": count} for word, count in keywords],
                    "completion_rates": completion_analysis
                },
                "insights": insights,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的关键词分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'keyword_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/length-analysis", summary="获取标题长度分析")
async def get_length_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取标题长度分析数据

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含标题长度分析的数据
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
            cached_response = pattern_cache.get_cached_patterns(table_name, 'length_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的标题长度分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的标题长度数据")

        # 获取标题长度分析
        length_analysis = analyze_title_length(cursor, table_name)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "length_analysis": length_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的标题长度分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'length_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/sentiment-analysis", summary="获取标题情感分析")
async def get_sentiment_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取标题情感分析数据

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含标题情感分析的数据
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
            cached_response = pattern_cache.get_cached_patterns(table_name, 'sentiment_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的标题情感分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的标题情感数据")

        # 获取标题情感分析
        sentiment_analysis = analyze_title_sentiment(cursor, table_name)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "sentiment_analysis": sentiment_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的标题情感分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'sentiment_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/trend-analysis", summary="获取标题趋势分析")
async def get_trend_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取标题趋势分析数据

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含标题趋势分析的数据
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
            cached_response = pattern_cache.get_cached_patterns(table_name, 'trend_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的标题趋势分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的标题趋势数据")

        # 获取标题趋势分析
        trend_analysis = analyze_title_trends(cursor, table_name)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "trend_analysis": trend_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的标题趋势分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'trend_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/interaction-analysis", summary="获取标题互动分析")
async def get_interaction_analysis(
    year: Optional[int] = Query(None, description="要分析的年份，不传则使用当前年份"),
    use_cache: bool = Query(True, description="是否使用缓存，默认为True。如果为False则重新分析数据")
):
    """获取标题互动分析数据

    Args:
        year: 要分析的年份，不传则使用当前年份
        use_cache: 是否使用缓存，默认为True。如果为False则重新分析数据

    Returns:
        dict: 包含标题互动分析的数据
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
            cached_response = pattern_cache.get_cached_patterns(table_name, 'interaction_analysis')
            if cached_response:
                print(f"从缓存获取 {target_year} 年的标题互动分析数据")
                return cached_response

        print(f"开始分析 {target_year} 年的标题互动数据")

        # 获取标题互动分析
        interaction_analysis = analyze_title_interaction(cursor, table_name)

        # 构建响应
        response = {
            "status": "success",
            "data": {
                "interaction_analysis": interaction_analysis,
                "year": target_year,
                "available_years": available_years
            }
        }

        # 更新缓存
        from .title_pattern_discovery import pattern_cache
        print(f"更新 {target_year} 年的标题互动分析数据缓存")
        pattern_cache.cache_patterns(table_name, 'interaction_analysis', response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
