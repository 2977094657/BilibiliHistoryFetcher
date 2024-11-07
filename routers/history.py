import json
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from scripts.utils import get_output_path, load_config

router = APIRouter()
config = load_config()

def get_db():
    """获取数据库连接"""
    db_path = get_output_path(config['db_file'])
    return sqlite3.connect(db_path)

@router.get("/all")
async def get_history_page(
    page: int = Query(1, description="当前页码"),
    size: int = Query(10, description="每页记录数"),
    sort_order: int = Query(0, description="排序顺序，0为降序，1为升序"),
    tag_name: Optional[str] = Query(None, description="视频子分区名称"),
    main_category: Optional[str] = Query(None, description="主分区名称"),
    date_range: Optional[str] = Query(None, description="日期范围，格式为yyyyMMdd-yyyyMMdd")
):
    """分页查询历史记录"""
    # 打印接收到的参数
    print("\n=== 接收到的请求参数 ===")
    print(f"页码(page): {page}")
    print(f"每页记录数(size): {size}")
    print(f"排序顺序(sort_order): {'升序' if sort_order == 1 else '降序'}")
    print(f"子分区名称(tag_name): {tag_name if tag_name else '无'}")
    print(f"主分区名称(main_category): {main_category if main_category else '无'}")
    print(f"日期范围(date_range): {date_range if date_range else '无'}")
    print("=====================\n")

    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # 构建基础查询
        current_year = datetime.now().year
        table_name = f"bilibili_history_{current_year}"
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = []

        # 打印SQL查询开始
        print("=== SQL查询构建 ===")
        print(f"基础查询: {query}")

        # 处理日期范围
        if date_range:
            try:
                start_date, end_date = date_range.split('-')
                start_timestamp = int(datetime.strptime(start_date, '%Y%m%d').timestamp())
                end_timestamp = int(datetime.strptime(end_date, '%Y%m%d').timestamp()) + 86400
                query += " AND view_at >= ? AND view_at < ?"
                params.extend([start_timestamp, end_timestamp])
                print(f"添加日期范围条件: {start_date} 到 {end_date}")
            except ValueError:
                print("日期格式错误")
                return {"status": "error", "message": "日期格式无效，应为yyyyMMdd-yyyyMMdd"}

        # 处理分类筛选
        if main_category:
            query += " AND main_category = ?"
            params.append(main_category)
            print(f"添加主分区筛选: {main_category}")
        elif tag_name:
            query += " AND tag_name = ?"
            params.append(tag_name)
            print(f"添加子分区筛选: {tag_name}")

        # 添加排序
        query += " ORDER BY view_at " + ("ASC" if sort_order == 1 else "DESC")
        print(f"添加排序: {'升序' if sort_order == 1 else '降序'}")

        # 获取总记录数
        count_query = f"SELECT COUNT(*) FROM ({query})"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        print(f"总记录数: {total}")

        # 添加分页
        query += " LIMIT ? OFFSET ?"
        params.extend([size, (page - 1) * size])
        print(f"添加分页: 第{page}页，每页{size}条")
        print(f"最终SQL: {query}")
        print(f"参数: {params}")
        print("==================\n")

        # 执行查询
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        records = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            if 'covers' in record and record['covers']:
                try:
                    record['covers'] = json.loads(record['covers'])
                except json.JSONDecodeError:
                    record['covers'] = []
            records.append(record)

        # 打印响应结果
        print("=== 响应结果 ===")
        print(f"返回记录数: {len(records)}")
        print(f"第一条记录: {records[0] if records else '无记录'}")
        print("================\n")

        return {
            "status": "success",
            "data": {
                "records": records,
                "total": total,
                "size": size,
                "current": page
            }
        }

    except sqlite3.Error as e:
        error_msg = f"数据库错误: {str(e)}"
        print(f"=== 错误 ===\n{error_msg}\n===========")
        return {"status": "error", "message": error_msg}
    finally:
        if conn:
            conn.close()

@router.get("/search")
async def search_history(
    page: int = Query(1, description="当前页码"),
    size: int = Query(10, description="每页记录数"),
    sortOrder: int = Query(0, description="排序顺序，0为降序，1为升序"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """搜索历史记录"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        current_year = datetime.now().year
        table_name = f"bilibili_history_{current_year}"
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = []

        if search:
            search = search.strip()
            if search.isdigit():
                # 按 oid 精确搜索
                query += " AND oid = ?"
                params.append(int(search))
            else:
                # 按标题模糊搜索
                query += " AND title LIKE ?"
                params.append(f"%{search}%")

        # 添加排序
        query += " ORDER BY view_at " + ("ASC" if sortOrder == 1 else "DESC")

        # 获取总记录数
        count_query = f"SELECT COUNT(*) FROM ({query})"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        # 添加分页
        query += " LIMIT ? OFFSET ?"
        params.extend([size, (page - 1) * size])

        # 执行查询
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        records = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            if 'covers' in record and record['covers']:
                try:
                    record['covers'] = json.loads(record['covers'])
                except json.JSONDecodeError:
                    record['covers'] = []
            records.append(record)

        return {
            "status": "success",
            "data": {
                "records": records,
                "total": total,
                "size": size,
                "current": page
            }
        }

    except sqlite3.Error as e:
        return {"status": "error", "message": f"数据库错误: {str(e)}"}
    finally:
        if conn:
            conn.close() 