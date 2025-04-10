import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TypedDict

# 配置日志
# 确保输出目录存在
os.makedirs("output/check", exist_ok=True)


# 设置自定义的日志格式化器，以正确处理中文字符
class EncodingFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        # 确保返回的是原始字符串，不会被转义
        if isinstance(msg, str):
            return msg
        else:
            return str(msg)


# 配置日志处理程序
formatter = EncodingFormatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    "output/check/data_integrity_check.log", mode="a", encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []  # 清除可能存在的处理程序
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # 防止日志消息传播到根日志器


class JsonInfo(TypedDict):
    path: Path
    year: int
    month: int
    day: int


def get_json_files(json_root_path: Path) -> list[JsonInfo]:
    """获取所有JSON文件的路径"""
    json_files = []

    if not json_root_path.exists():
        return []

    def is_valid_dir(dir: Path) -> bool:
        return dir.is_dir() and dir.stem.isdigit()

    for year_dir in json_root_path.iterdir():
        if not is_valid_dir(year_dir):
            continue
        for month_dir in year_dir.iterdir():
            if not is_valid_dir(month_dir):
                continue
            for day_file in month_dir.iterdir():
                if day_file.suffix != ".json":
                    continue
                json_files.append(
                    {
                        "path": day_file,
                        "year": int(year_dir.stem),
                        "month": int(month_dir.stem),
                        "day": int(day_file.stem),
                    }
                )

    return json_files


def count_records_in_json_file(file_path):
    """统计JSON文件中的记录数量，并返回所有记录标题"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            titles = [item.get("title", "未知标题") for item in data]
            return len(data), titles
    except Exception as e:
        logger.error(f"读取JSON文件 {file_path} 时出错: {e}")
        return 0, []


def get_db_tables(db_path):
    """获取数据库中的所有表名"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        logger.error(f"获取数据库表时出错: {e}")
        return []


def count_records_in_db_table(db_path, table_name):
    """统计数据库表中的记录数量"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"统计表 {table_name} 记录数时出错: {e}")
        return 0


def get_records_by_date(db_path, table_name, year, month, day):
    """获取某一天的数据库记录"""
    try:
        # 计算目标日期的时间戳范围
        start_date = datetime(year, month, day).timestamp()
        end_date = datetime(year, month, day, 23, 59, 59).timestamp()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT title, view_at FROM {table_name} WHERE view_at >= ? AND view_at <= ?",
            (start_date, end_date),
        )
        records = cursor.fetchall()
        titles = [record[0] for record in records]
        conn.close()
        return len(records), titles
    except Exception as e:
        logger.error(f"获取{year}年{month}月{day}日的记录时出错: {e}")
        return 0, []


def check_data_integrity(db_path=None, json_root_path=None):
    """检查数据完整性"""
    output_dir = Path("output")
    # 配置路径
    if db_path is None:
        db_path = output_dir / "bilibili_history.db"
    if json_root_path is None:
        json_root_path = output_dir / "history_by_date"

    logger.info("开始数据完整性检查...")
    logger.info(f"数据库路径: {db_path}")
    logger.info(f"JSON文件路径: {json_root_path}")

    # 创建输出目录
    output_dir = os.path.join("output", "check")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "total_json_files": 0,
        "total_json_records": 0,
        "total_db_records": 0,
        "db_tables": [],
        "missing_records": [],
        "extra_records": [],
    }

    # 获取数据库表
    tables = get_db_tables(db_path)
    results["db_tables"] = tables
    logger.info(f"数据库包含以下表: {', '.join(tables)}")

    # 获取历史表（以bilibili_history_开头的表）
    history_tables = [
        table for table in tables if table.startswith("bilibili_history_")
    ]

    # 遍历所有JSON文件
    json_info = get_json_files(json_root_path)
    results["total_json_files"] = len(json_info)
    logger.info(f"找到 {len(json_info)} 个JSON文件")

    all_json_records = 0
    all_db_records = 0

    for file_info in json_info:
        file_path = file_info["path"]
        year = file_info["year"]
        month = file_info["month"]
        day = file_info["day"]

        # 统计JSON文件中的记录
        json_count, json_titles = count_records_in_json_file(file_path)
        all_json_records += json_count

        if json_count == 0:
            logger.warning(f"JSON文件为空: {file_path}")
            continue

        # 查找对应年份的数据库表
        table_name = f"bilibili_history_{year}"
        if table_name not in history_tables:
            logger.error(f"数据库中缺少表 {table_name}")
            results["missing_records"].append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "missing_count": json_count,
                    "missing_titles": json_titles,
                    "reason": f"数据库中缺少表 {table_name}",
                }
            )
            continue

        # 获取数据库中对应日期的记录
        db_count, db_titles = get_records_by_date(db_path, table_name, year, month, day)
        all_db_records += db_count

        # 比较记录数量
        if json_count > db_count:
            missing_count = json_count - db_count
            # 找出缺少的标题
            missing_titles = [title for title in json_titles if title not in db_titles]

            results["missing_records"].append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "missing_count": missing_count,
                    "missing_titles": missing_titles[:10]
                    if len(missing_titles) > 10
                    else missing_titles,  # 最多显示10个
                    "reason": "数据库记录少于JSON文件",
                }
            )
            logger.warning(
                f"{year}年{month}月{day}日 - 数据库中缺少 {missing_count} 条记录"
            )

        elif json_count < db_count:
            extra_count = db_count - json_count
            # 找出多余的标题
            extra_titles = [title for title in db_titles if title not in json_titles]

            results["extra_records"].append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "extra_count": extra_count,
                    "extra_titles": extra_titles[:10]
                    if len(extra_titles) > 10
                    else extra_titles,  # 最多显示10个
                    "reason": "数据库记录多于JSON文件",
                }
            )
            logger.warning(
                f"{year}年{month}月{day}日 - 数据库中多出 {extra_count} 条记录"
            )

    # 统计总记录数
    results["total_json_records"] = all_json_records
    results["total_db_records"] = all_db_records

    for table in history_tables:
        table_count = count_records_in_db_table(db_path, table)
        logger.info(f"表 {table} 中有 {table_count} 条记录")

    if all_json_records > all_db_records:
        logger.error(f"总缺少记录数: {all_json_records - all_db_records}")
    elif all_json_records < all_db_records:
        logger.error(f"总多余记录数: {all_db_records - all_json_records}")
    else:
        logger.info("JSON文件和数据库中的记录数量完全匹配")

    # 保存结果到JSON文件
    result_file = os.path.join(output_dir, "data_integrity_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"检查完成，结果已保存到 {result_file}")

    # 生成报告
    report_file = generate_report(results)

    return {
        "success": True,
        "result_file": result_file,
        "report_file": report_file,
        "total_json_files": results["total_json_files"],
        "total_json_records": results["total_json_records"],
        "total_db_records": results["total_db_records"],
        "missing_records_count": len(results["missing_records"]),
        "extra_records_count": len(results["extra_records"]),
        "difference": all_json_records - all_db_records,
    }


def generate_report(results):
    """生成报告"""
    # 创建输出目录
    output_dir = os.path.join("output", "check")
    os.makedirs(output_dir, exist_ok=True)

    report = ["# 数据完整性检查报告\n"]

    # 基本信息
    report.append("## 基本信息\n")
    report.append(f"* 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"* JSON文件总数: {results['total_json_files']}")
    report.append(f"* JSON记录总数: {results['total_json_records']}")
    report.append(f"* 数据库记录总数: {results['total_db_records']}")
    report.append(f"* 数据库表: {', '.join(results['db_tables'])}\n")

    # 总体差异
    diff = results["total_json_records"] - results["total_db_records"]
    if diff > 0:
        report.append(f"## 总体情况: 数据库缺少 {diff} 条记录\n")
    elif diff < 0:
        report.append(f"## 总体情况: 数据库多出 {-diff} 条记录\n")
    else:
        report.append("## 总体情况: 数据库和JSON文件记录数一致\n")

    # 缺少的记录
    if results["missing_records"]:
        report.append(f"## 缺少的记录 (共 {len(results['missing_records'])} 天)\n")
        for item in results["missing_records"]:
            report.append(
                f"### {item['year']}年{item['month']}月{item['day']}日 - 缺少 {item['missing_count']} 条记录"
            )
            report.append(f"原因: {item['reason']}")
            if item["missing_titles"]:
                report.append("缺少的标题示例:")
                for title in item["missing_titles"]:
                    report.append(f"* {title}")
            report.append("")

    # 多余的记录
    if results["extra_records"]:
        report.append(f"## 多余的记录 (共 {len(results['extra_records'])} 天)\n")
        for item in results["extra_records"]:
            report.append(
                f"### {item['year']}年{item['month']}月{item['day']}日 - 多出 {item['extra_count']} 条记录"
            )
            report.append(f"原因: {item['reason']}")
            if item["extra_titles"]:
                report.append("多余的标题示例:")
                for title in item["extra_titles"]:
                    report.append(f"* {title}")
            report.append("")

    # 保存报告
    report_file = os.path.join(output_dir, "data_integrity_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    logger.info(f"报告已保存到 {report_file}")
    return report_file


def main():
    db_path = os.path.join("output", "bilibili_history.db")
    json_root_path = os.path.join("output", "history_by_date")

    print("=" * 50)
    print("开始检查数据完整性...")
    print(f"数据库路径: {db_path}")
    print(f"JSON文件路径: {json_root_path}")
    print("=" * 50)

    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return

    if not os.path.exists(json_root_path):
        print(f"错误: JSON文件目录不存在: {json_root_path}")
        return

    results = check_data_integrity(db_path, json_root_path)

    print("\n" + "=" * 50)
    print(f"检查完成! 详细结果已保存到 {results['report_file']}")
    print("=" * 50)

    # 简要报告
    diff = results["difference"]
    if diff > 0:
        print(f"数据库总共缺少 {diff} 条记录")
        if results["missing_records_count"] > 0:
            print(f"\n缺少记录的日期数: {results['missing_records_count']}")
    elif diff < 0:
        print(f"数据库总共多出 {-diff} 条记录")
    else:
        print("数据库和JSON文件记录数一致")


if __name__ == "__main__":
    main()
