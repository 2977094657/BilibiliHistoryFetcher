import json
import os
import sqlite3
import time
from typing import Optional, List, Dict, Any, Union

import requests
import yaml
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, Field

from scripts.utils import get_output_path, load_config
from scripts.wbi_sign import get_wbi_sign

# 导入DeepSeek API相关模块
from routers.deepseek import chat_completion, ChatMessage, ChatRequest

router = APIRouter()
config = load_config()

# 获取是否缓存空摘要的配置，默认为True
CACHE_EMPTY_SUMMARY = config.get("CACHE_EMPTY_SUMMARY", True)


# 定义配置模型
class SummaryConfig(BaseModel):
    cache_empty_summary: bool = True


class VideoSummaryPrompt(BaseModel):
    default_prompt: str = Field(..., description="默认提示词，用于重置")
    custom_prompt: str = Field(..., description="用户自定义提示词")


# 数据模型定义
class OutlinePoint(BaseModel):
    timestamp: int
    content: str


class OutlinePart(BaseModel):
    title: str
    part_outline: List[OutlinePoint]
    timestamp: int


class VideoSummaryResponse(BaseModel):
    bvid: str
    cid: int
    up_mid: int
    stid: Optional[str] = None
    summary: Optional[str] = None
    outline: Optional[List[OutlinePart]] = None
    result_type: int = 0
    status_message: str = ""  # 用于解释result_type的含义
    has_summary: bool = False  # 表示是否存在摘要内容
    fetch_time: int
    update_time: Optional[int] = None
    from_cache: bool = False


class SummaryRequest(BaseModel):
    bvid: str
    cid: int
    up_mid: int


def get_db():
    """获取数据库连接"""
    db_path = get_output_path(config["db_file"])

    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 设置数据库兼容性参数
        pragmas = [
            ("legacy_file_format", 1),
            ("journal_mode", "DELETE"),
            ("synchronous", "NORMAL"),
            ("user_version", 317),  # 使用固定的用户版本号
            ("encoding", "UTF8"),  # 设置数据库编码为UTF8 (不使用连字符)
        ]

        for pragma, value in pragmas:
            cursor.execute(f"PRAGMA {pragma}={value}")
        conn.commit()

        # 确保视频摘要表存在
        from config.sql_statements_sqlite import (
            CREATE_TABLE_VIDEO_SUMMARY,
            CREATE_INDEXES_VIDEO_SUMMARY,
        )

        cursor.execute(CREATE_TABLE_VIDEO_SUMMARY)

        # 创建索引
        for index_sql in CREATE_INDEXES_VIDEO_SUMMARY:
            cursor.execute(index_sql)

        conn.commit()

        return conn

    except sqlite3.Error as e:
        print(f"数据库连接错误: {str(e)}")
        if "conn" in locals() and conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"数据库连接失败: {str(e)}")


def get_video_summary_from_db(bvid: str, cid: int) -> Optional[Dict[str, Any]]:
    """从数据库获取视频摘要"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT bvid, cid, up_mid, stid, summary, outline, result_type, fetch_time, update_time
            FROM video_summary
            WHERE bvid = ? AND cid = ?
        """,
            (bvid, cid),
        )

        result = cursor.fetchone()
        if result:
            result_type = result[6]
            status_message = get_status_message(result_type)

            # 判断是否有有效摘要 (result_type为1或2表示有摘要)
            has_summary = result_type > 0

            return {
                "bvid": result[0],
                "cid": result[1],
                "up_mid": result[2],
                "stid": result[3],
                "summary": result[4],
                "outline": json.loads(result[5]) if result[5] else None,
                "result_type": result_type,
                "status_message": status_message,
                "has_summary": has_summary,
                "fetch_time": result[7],
                "update_time": result[8],
                "from_cache": True,
            }
        return None
    except sqlite3.Error as e:
        print(f"查询数据库时发生错误: {e}")
        return None
    finally:
        if conn:
            conn.close()


# 添加一个辅助函数获取状态消息
def get_status_message(result_type: int) -> str:
    """
    根据result_type获取对应的状态消息

    官方状态码含义：
    - 0: 没有摘要
    - 1: 仅存在摘要总结
    - 2: 存在摘要以及提纲
    """
    if result_type == 0:
        return "该视频没有摘要"
    elif result_type == 1:
        return "该视频仅有摘要总结"
    elif result_type == 2:
        return "该视频有摘要总结和提纲"
    elif result_type == -1:
        return "该视频不支持AI摘要（可能含有敏感内容或其他因素导致）"
    else:
        return f"未知状态码: {result_type}"


def save_video_summary_to_db(
    bvid: str,
    cid: int,
    up_mid: int,
    stid: str,
    summary: str,
    outline: Union[List[Dict], None],
    result_type: int,
) -> bool:
    """保存视频摘要到数据库"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        current_time = int(time.time())

        # 检查是否已存在
        cursor.execute(
            "SELECT id FROM video_summary WHERE bvid = ? AND cid = ?", (bvid, cid)
        )

        existing = cursor.fetchone()

        if existing:
            # 更新现有记录
            from config.sql_statements_sqlite import UPDATE_VIDEO_SUMMARY

            cursor.execute(
                UPDATE_VIDEO_SUMMARY,
                (
                    stid,
                    summary,
                    json.dumps(outline, ensure_ascii=False) if outline else None,
                    result_type,
                    current_time,
                    bvid,
                    cid,
                ),
            )
        else:
            # 插入新记录
            from config.sql_statements_sqlite import INSERT_VIDEO_SUMMARY

            # 生成ID (使用时间戳+随机数)
            id = int(f"{current_time}{hash(bvid + str(cid)) % 10000:04d}")

            cursor.execute(
                INSERT_VIDEO_SUMMARY,
                (
                    id,
                    bvid,
                    cid,
                    up_mid,
                    stid,
                    summary,
                    json.dumps(outline, ensure_ascii=False) if outline else None,
                    result_type,
                    current_time,
                    current_time,
                ),
            )

        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"保存到数据库时发生错误: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


async def fetch_video_summary_from_api(
    bvid: str, cid: int, up_mid: int
) -> Dict[str, Any]:
    """从B站API获取视频摘要"""
    try:
        # 获取wbi签名
        wbi_params = {"bvid": bvid, "cid": cid, "up_mid": up_mid}

        signed_params = get_wbi_sign(wbi_params)

        # 构建请求URL
        url = "https://api.bilibili.com/x/web-interface/view/conclusion/get"

        # 从配置中读取SESSDATA
        sessdata = config.get("SESSDATA", "")

        # 添加必要的HTTP头信息
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
        }

        # 如果有SESSDATA，添加到Cookie
        if sessdata:
            headers["Cookie"] = f"SESSDATA={sessdata}"

        # 发送请求
        response = requests.get(url, params=signed_params, headers=headers)
        data = response.json()

        # 记录原始返回数据用于调试
        print(f"API返回数据: {json.dumps(data, ensure_ascii=False)}")

        # 根据返回的code进行处理
        if data["code"] == 0:
            # API请求成功
            return data["data"]
        elif data["code"] == -1:
            # 不支持AI摘要或请求异常
            return {
                "model_result": {
                    "result_type": -1,  # 使用-1表示不支持AI摘要
                    "summary": "该视频不支持AI摘要（可能含有敏感内容或其他因素导致）",
                    "outline": None,
                },
                "stid": "",
            }
        else:
            # 其他错误情况
            raise HTTPException(
                status_code=400,
                detail=f"获取视频摘要失败: {data['message']} (code: {data['code']})",
            )

    except Exception as e:
        # 记录详细错误信息
        print(f"获取视频摘要时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取视频摘要时发生错误: {str(e)}")


@router.get("/get_summary", summary="获取视频摘要", response_model=VideoSummaryResponse)
async def get_video_summary(
    bvid: str, cid: int, up_mid: int, force_refresh: Optional[bool] = False
):
    """
    获取视频摘要

    - **bvid**: 视频的BVID
    - **cid**: 视频的CID
    - **up_mid**: UP主的MID
    - **force_refresh**: 是否强制刷新（不使用缓存）
    """
    try:
        # 首先尝试从数据库获取
        if not force_refresh:
            db_result = get_video_summary_from_db(bvid, cid)
            if db_result:
                return db_result

        # 从API获取
        api_result = await fetch_video_summary_from_api(bvid, cid, up_mid)

        # 解析API结果
        result_type = api_result.get("model_result", {}).get("result_type", 0)
        summary = api_result.get("model_result", {}).get("summary", "")
        outline_data = api_result.get("model_result", {}).get("outline", None)
        stid = api_result.get("stid", "")

        # 获取状态消息
        status_message = get_status_message(result_type)

        # 判断是否有有效摘要 (result_type为1或2表示有摘要)
        has_summary = result_type > 0

        # 根据配置决定是否保存到数据库
        # 如果CACHE_EMPTY_SUMMARY为True，则保存所有结果
        # 如果CACHE_EMPTY_SUMMARY为False，则只保存有摘要的结果
        should_save = has_summary or CACHE_EMPTY_SUMMARY

        if should_save:
            # 保存到数据库
            save_success = save_video_summary_to_db(
                bvid=bvid,
                cid=cid,
                up_mid=up_mid,
                stid=stid,
                summary=summary,
                outline=outline_data,
                result_type=result_type,
            )
            if not save_success:
                print(f"警告: 保存视频摘要到数据库失败: {bvid}, {cid}")
        else:
            print(
                f"跳过保存无摘要数据到数据库: {bvid}, {cid}, result_type={result_type}"
            )

        # 保存B站获取的摘要到./output/BSummary/{cid}目录
        # 不管是否有摘要内容，都保存，因为判断太耗时间
        try:
            # 创建保存目录
            save_dir = os.path.join("output", "BSummary", str(cid))
            os.makedirs(save_dir, exist_ok=True)

            # 构建完整的响应数据
            response_data = {
                "bvid": bvid,
                "cid": cid,
                "up_mid": up_mid,
                "stid": stid,
                "summary": summary,
                "outline": outline_data,
                "result_type": result_type,
                "status_message": status_message,
                "has_summary": has_summary,
                "fetch_time": int(time.time()),
                "update_time": int(time.time()),
                "from_cache": False,
                "api_response": api_result,  # 保存原始API响应
            }

            # 保存完整响应数据
            response_path = os.path.join(save_dir, f"{cid}_response.json")
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            # 如果有摘要，单独保存摘要内容到文本文件，方便查看
            if has_summary:
                summary_path = os.path.join(save_dir, f"{cid}_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)

                # 如果有提纲，单独保存提纲
                if outline_data:
                    outline_path = os.path.join(save_dir, f"{cid}_outline.json")
                    with open(outline_path, "w", encoding="utf-8") as f:
                        json.dump(outline_data, f, ensure_ascii=False, indent=2)

            print(f"已保存B站摘要到: {save_dir}")
        except Exception as e:
            # 保存到文件失败不影响API返回
            print(f"警告: 保存B站摘要到文件失败: {str(e)}")

        # 返回结果
        return {
            "bvid": bvid,
            "cid": cid,
            "up_mid": up_mid,
            "stid": stid,
            "summary": summary,
            "outline": outline_data,
            "result_type": result_type,
            "status_message": status_message,
            "has_summary": has_summary,
            "fetch_time": int(time.time()),
            "update_time": int(time.time()),
            "from_cache": False,
        }

    except Exception as e:
        # 捕获所有可能的异常，确保API有良好的错误处理
        print(f"获取视频摘要出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取视频摘要失败: {str(e)}")


# 配置操作函数
def save_config(new_config: Dict[str, Any]) -> bool:
    """保存配置到配置文件，只修改指定的配置项，保持文件其余部分不变"""
    try:
        # 获取配置文件路径
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "config.yaml",
        )

        # 读取配置文件内容
        with open(config_path, "r", encoding="utf-8") as f:
            config_content = f.read()

        # 只更新指定的字段
        for key, value in new_config.items():
            # 对于布尔值，需要特殊处理
            if isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)

            # 使用正则表达式查找并替换配置项
            import re

            pattern = rf"^{key}\s*:.*$"
            replacement = f"{key}: {value_str}"
            config_content = re.sub(
                pattern, replacement, config_content, flags=re.MULTILINE
            )

        # 写回配置文件
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        # 更新全局变量
        global config, CACHE_EMPTY_SUMMARY
        config.update(new_config)  # 只更新指定的字段
        CACHE_EMPTY_SUMMARY = config.get("CACHE_EMPTY_SUMMARY", True)

        return True
    except Exception as e:
        print(f"保存配置失败: {str(e)}")
        return False


# 添加配置接口
@router.get("/config", summary="获取摘要相关配置", response_model=SummaryConfig)
async def get_summary_config():
    """获取摘要相关配置"""
    return {"cache_empty_summary": CACHE_EMPTY_SUMMARY}


@router.post("/config", summary="更新摘要相关配置", response_model=SummaryConfig)
async def update_summary_config(config_data: SummaryConfig = Body(...)):
    """
    更新摘要相关配置

    - **cache_empty_summary**: 是否缓存空摘要结果
    """
    # 准备要更新的配置项，只包含CACHE_EMPTY_SUMMARY字段
    config_updates = {"CACHE_EMPTY_SUMMARY": config_data.cache_empty_summary}

    try:
        # 读取当前配置
        config_path = get_output_path("config/config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f)

        # 更新配置
        current_config.update(config_updates)

        # 写回配置文件
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)

        # 更新全局配置
        global config, CACHE_EMPTY_SUMMARY
        config = load_config()
        CACHE_EMPTY_SUMMARY = config.get("CACHE_EMPTY_SUMMARY", True)

        return {"cache_empty_summary": CACHE_EMPTY_SUMMARY}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


class LLMBaseRequest(BaseModel):
    model: str = Field(default="deepseek-chat", description="DeepSeek模型名称")
    temperature: float = Field(default=0.7, description="生成温度，控制创造性")
    max_tokens: int = Field(default=1000, description="最大生成标记数")
    frequency_penalty: float = Field(
        default=0,
        description="频率惩罚，避免重复",
    )
    presence_penalty: float = Field(
        default=0,
        description="存在惩罚，增加话题多样性",
    )

    top_p: float = Field(
        default=1,
        description="核采样阈值，控制输出的多样性",
    )
    stream: bool = Field(default=False, description="是否使用流式输出")
    json_mode: bool = Field(default=False, description="是否使用JSON模式输出")


# 自定义视频摘要请求模型
class CustomSummaryRequest(LLMBaseRequest):
    bvid: str
    cid: int
    up_mid: int
    subtitle_content: str = Field(..., description="视频字幕内容")


# 添加 TokenDetails 和 UsageInfo 模型
class TokenDetails(BaseModel):
    cached_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[TokenDetails] = None


# 自定义视频摘要响应模型
class CustomSummaryResponse(BaseModel):
    bvid: str
    cid: int
    up_mid: int
    summary: str
    success: bool
    message: str
    from_deepseek: bool = True
    tokens_used: Optional[UsageInfo] = None
    processing_time: Optional[float] = None


# 使用DeepSeek API生成自定义视频摘要
async def generate_custom_summary(
    subtitle_content: str,
    background_tasks: BackgroundTasks,
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 8000,
    top_p: float = 1.0,
    stream: bool = False,
    json_mode: bool = False,
) -> Dict[str, Any]:
    """生成自定义视频摘要"""
    try:
        # 加载配置获取提示词
        config = load_config()
        prompt = (
            config.get("deepseek", {}).get("video_summary", {}).get("custom_prompt", "")
        )

        if not prompt:
            raise ValueError("未找到有效的提示词配置")

        # 构建消息列表
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": subtitle_content},
        ]

        # 调用DeepSeek API
        start_time = time.time()
        response = await chat_completion(
            request=ChatRequest(
                messages=[ChatMessage(**msg) for msg in messages],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                json_mode=json_mode,
            ),
            background_tasks=background_tasks,
        )
        processing_time = time.time() - start_time

        # 构造符合 UsageInfo 模型的 tokens_used 数据
        usage = response.get("usage", {})
        tokens_used = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "prompt_tokens_details": {
                "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                    "cached_tokens", 0
                )
            }
            if usage.get("prompt_tokens_details")
            else None,
        }

        return {
            "success": True,
            "summary": response["content"],
            "tokens_used": tokens_used,
            "processing_time": processing_time,
        }
    except Exception as e:
        return {"success": False, "message": f"生成摘要失败: {str(e)}"}


@router.post(
    "/custom_summary",
    summary="使用DeepSeek生成自定义视频摘要",
    response_model=CustomSummaryResponse,
)
async def create_custom_summary(
    request: CustomSummaryRequest, background_tasks: BackgroundTasks
):
    """
    使用DeepSeek API生成自定义视频摘要

    - **bvid**: 视频的BVID
    - **cid**: 视频的CID
    - **up_mid**: UP主的MID
    - **subtitle_content**: 视频字幕内容
    - **model**: DeepSeek模型名称，默认为deepseek-chat
    - **temperature**: 生成温度，控制创造性，默认为0.7
    - **max_tokens**: 最大生成标记数，默认为1000
    """
    try:
        # 生成自定义摘要
        result = await generate_custom_summary(
            subtitle_content=request.subtitle_content,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            background_tasks=background_tasks,
        )

        # 如果生成成功且配置允许，保存到数据库
        if result["success"] and result["summary"] and CACHE_EMPTY_SUMMARY:
            try:
                # 保存到数据库
                save_success = save_video_summary_to_db(
                    bvid=request.bvid,
                    cid=request.cid,
                    up_mid=request.up_mid,
                    stid="custom_deepseek",  # 使用特殊标识表示这是自定义摘要
                    summary=result["summary"],
                    outline=None,  # 自定义摘要暂不支持提纲
                    result_type=1,  # 使用1表示有摘要但无提纲
                )
                if not save_success:
                    print(
                        f"警告: 保存自定义视频摘要到数据库失败: {request.bvid}, {request.cid}"
                    )
            except Exception as e:
                print(f"保存自定义摘要到数据库时出错: {str(e)}")

        # 构建并返回响应
        return {
            "bvid": request.bvid,
            "cid": request.cid,
            "up_mid": request.up_mid,
            "summary": result["summary"],
            "success": result["success"],
            "message": result.get("message", ""),
            "from_deepseek": True,
            "tokens_used": result.get("tokens_used"),
            "processing_time": result.get("processing_time"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"生成自定义摘要时发生错误: {str(e)}"
        )


# 从字幕文件生成摘要的请求模型
class SubtitleFileRequest(BaseModel):
    bvid: str
    cid: int
    up_mid: int
    subtitle_file: str = Field(..., description="字幕文件路径，支持SRT或JSON格式")
    model: str = Field(default="deepseek-chat", description="DeepSeek模型名称")
    temperature: float = Field(default=0.7, description="生成温度，控制创造性")
    max_tokens: int = Field(default=1000, description="最大生成标记数")


@router.post(
    "/summarize_from_subtitle",
    summary="从字幕文件生成视频摘要",
    response_model=CustomSummaryResponse,
)
async def summarize_from_subtitle(
    request: SubtitleFileRequest, background_tasks: BackgroundTasks
):
    """
    从字幕文件生成视频摘要

    - **bvid**: 视频的BVID
    - **cid**: 视频的CID
    - **up_mid**: UP主的MID
    - **subtitle_file**: 字幕文件路径，支持SRT或JSON格式
    - **model**: DeepSeek模型名称，默认为deepseek-chat
    - **temperature**: 生成温度，控制创造性，默认为0.7
    - **max_tokens**: 最大生成标记数，默认为1000
    """
    try:
        # 检查字幕文件是否存在
        if not os.path.exists(request.subtitle_file):
            raise HTTPException(
                status_code=400, detail=f"字幕文件不存在: {request.subtitle_file}"
            )

        # 读取字幕文件内容
        subtitle_content = ""
        file_ext = os.path.splitext(request.subtitle_file)[1].lower()

        if file_ext == ".srt":
            # 处理SRT格式
            with open(request.subtitle_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 提取字幕文本，忽略时间戳和序号
            current_text = ""
            for line in lines:
                line = line.strip()
                # 跳过空行、序号行和时间戳行
                if not line or line.isdigit() or "-->" in line:
                    continue
                current_text += line + " "

            subtitle_content = current_text

        elif file_ext == ".json":
            # 处理JSON格式
            with open(request.subtitle_file, "r", encoding="utf-8") as f:
                subtitle_data = json.load(f)

            # 根据JSON结构提取字幕文本
            if isinstance(subtitle_data, list):
                # 假设是包含字幕段落的列表
                for item in subtitle_data:
                    if isinstance(item, dict) and "content" in item:
                        subtitle_content += item["content"] + " "
            elif isinstance(subtitle_data, dict) and "body" in subtitle_data:
                # B站字幕格式
                for item in subtitle_data["body"]:
                    if "content" in item:
                        subtitle_content += item["content"] + " "
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的字幕文件格式: {file_ext}，仅支持.srt和.json",
            )

        if not subtitle_content:
            raise HTTPException(status_code=400, detail="无法从字幕文件中提取文本内容")

        # 创建自定义摘要请求
        summary_request = CustomSummaryRequest(
            bvid=request.bvid,
            cid=request.cid,
            up_mid=request.up_mid,
            subtitle_content=subtitle_content,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 调用自定义摘要生成函数
        return await create_custom_summary(summary_request, background_tasks)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"从字幕文件生成摘要时发生错误: {str(e)}"
        )


@router.get(
    "/prompt", summary="获取视频摘要提示词配置", response_model=VideoSummaryPrompt
)
async def get_summary_prompt():
    """获取视频摘要提示词配置"""
    try:
        config = load_config()
        prompt_config = config.get("deepseek", {}).get("video_summary", {})
        return VideoSummaryPrompt(
            default_prompt=prompt_config.get("default_prompt", ""),
            custom_prompt=prompt_config.get("custom_prompt", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取提示词配置失败: {str(e)}")


@router.post("/prompt", summary="更新视频摘要提示词", response_model=VideoSummaryPrompt)
async def update_summary_prompt(prompt: str = Body(..., description="新的提示词")):
    """更新视频摘要提示词配置"""
    try:
        config = load_config()

        # 将多行文本转换为单行（替换换行符为\n）
        prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")
        prompt = prompt.replace("\n", "\\n")

        # 更新配置
        if "deepseek" not in config:
            config["deepseek"] = {}
        if "video_summary" not in config["deepseek"]:
            config["deepseek"]["video_summary"] = {}

        # 保持默认提示词不变，只更新自定义提示词
        config["deepseek"]["video_summary"]["custom_prompt"] = prompt

        # 保存配置
        if not save_config(config):
            raise HTTPException(status_code=500, detail="保存配置失败")

        return VideoSummaryPrompt(
            default_prompt=config["deepseek"]["video_summary"]["default_prompt"],
            custom_prompt=prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新提示词配置失败: {str(e)}")


@router.post(
    "/prompt/reset",
    summary="重置视频摘要提示词到默认值",
    response_model=VideoSummaryPrompt,
)
async def reset_summary_prompt():
    """重置视频摘要提示词到默认值"""
    try:
        config = load_config()

        if "deepseek" not in config:
            config["deepseek"] = {}
        if "video_summary" not in config["deepseek"]:
            config["deepseek"]["video_summary"] = {}

        # 将自定义提示词重置为默认提示词
        default_prompt = config["deepseek"]["video_summary"]["default_prompt"]
        config["deepseek"]["video_summary"]["custom_prompt"] = default_prompt

        # 保存配置
        if not save_config(config):
            raise HTTPException(status_code=500, detail="保存配置失败")

        return VideoSummaryPrompt(
            default_prompt=default_prompt, custom_prompt=default_prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置提示词配置失败: {str(e)}")


# 添加新的请求模型
class CidSummaryRequest(LLMBaseRequest):
    cid: int = Field(..., description="视频的CID")


@router.post(
    "/summarize_by_cid",
    summary="根据CID生成视频摘要",
    response_model=CustomSummaryResponse,
)
async def summarize_by_cid(
    request: CidSummaryRequest, background_tasks: BackgroundTasks
):
    """根据CID生成视频摘要

    参数:
    - **cid**: 视频的CID
    - **model**: DeepSeek模型名称，默认为deepseek-chat
    - **temperature**: 生成温度，控制创造性，默认为0.7
    - **max_tokens**: 最大生成标记数，默认为1000
    """
    # 构建字幕文件路径
    stt_dir = os.path.join("output", "stt", str(request.cid))
    json_path = os.path.join(stt_dir, f"{request.cid}.json")

    # 检查字幕文件是否存在
    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=404, detail=f"未找到CID {request.cid} 的字幕文件"
        )

    # 读取字幕文件 - 直接作为文本读取，不尝试解析为JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            subtitle_content = f.read().strip()

        # 如果内容为空，抛出错误
        if not subtitle_content:
            raise HTTPException(status_code=400, detail="字幕文件内容为空")

        # 构造一个简单的元数据字典，用于返回结果
        subtitle_data = {
            "bvid": "",  # 这些字段可能在纯文本文件中不存在
            "up_mid": 0,
        }
    except json.JSONDecodeError:
        # 如果JSON解析失败，直接读取为文本
        with open(json_path, "r", encoding="utf-8") as f:
            subtitle_content = f.read().strip()

        # 构造一个简单的元数据字典
        subtitle_data = {"bvid": "", "up_mid": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取字幕文件时出错: {str(e)}")

    # 生成摘要
    result = await generate_custom_summary(
        subtitle_content=subtitle_content,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=request.stream,
        json_mode=request.json_mode,
        background_tasks=background_tasks,
    )

    # 构建响应数据
    response_data = {
        "bvid": subtitle_data.get("bvid", ""),
        "cid": request.cid,
        "up_mid": subtitle_data.get("up_mid", 0),
        "summary": result.get("summary", ""),
        "success": result.get("success", False),
        "message": result.get("message", ""),
        "from_deepseek": True,
        "tokens_used": result.get("tokens_used"),
        "processing_time": result.get("processing_time"),
    }

    # 保存摘要内容到文件
    if result.get("success", False) and result.get("summary"):
        try:
            # 创建保存目录
            summary_dir = os.path.join("output", "summary", str(request.cid))
            os.makedirs(summary_dir, exist_ok=True)

            # 保存摘要内容
            summary_path = os.path.join(summary_dir, f"{request.cid}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(result.get("summary", ""))

            # 保存完整响应数据（包括token使用情况等）
            response_path = os.path.join(summary_dir, f"{request.cid}_response.json")
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)

            print(f"摘要已保存到: {summary_path}")
            print(f"完整响应已保存到: {response_path}")
        except Exception as e:
            print(f"保存摘要时出错: {str(e)}")
            # 不影响正常返回，只记录错误

    return response_data


# 添加新的响应模型，用于检查本地摘要
class LocalSummaryCheckResponse(BaseModel):
    cid: int
    exists: bool
    summary_path: Optional[str] = None
    response_path: Optional[str] = None
    full_response: Optional[Dict[str, Any]] = None
    message: str


@router.get(
    "/check_local_summary/{cid}",
    summary="检查本地是否存在摘要文件",
    response_model=LocalSummaryCheckResponse,
)
async def check_local_summary(cid: int, include_content: bool = True):
    """
    检查本地是否存在指定CID的摘要文件

    参数:
    - **cid**: 视频的CID
    - **include_content**: 是否包含完整响应数据，默认为True

    返回:
    - **exists**: 是否存在摘要文件
    - **summary_path**: 摘要文件路径（如果存在）
    - **response_path**: 响应数据文件路径（如果存在）
    - **full_response**: 完整响应数据（如果存在且include_content=True）
    - **message**: 提示信息
    """
    try:
        # 构建摘要文件路径
        summary_dir = os.path.join("output", "summary", str(cid))
        summary_path = os.path.join(summary_dir, f"{cid}_summary.txt")
        response_path = os.path.join(summary_dir, f"{cid}_response.json")

        # 检查文件是否存在
        summary_exists = os.path.exists(summary_path)
        response_exists = os.path.exists(response_path)

        # 如果两个文件都不存在，返回不存在的响应
        if not summary_exists and not response_exists:
            return {
                "cid": cid,
                "exists": False,
                "summary_path": None,
                "response_path": None,
                "full_response": None,
                "message": f"未找到CID {cid} 的本地摘要文件",
            }

        # 构建响应数据
        result = {
            "cid": cid,
            "exists": True,
            "summary_path": summary_path if summary_exists else None,
            "response_path": response_path if response_exists else None,
            "full_response": None,
            "message": f"找到CID {cid} 的本地摘要文件",
        }

        # 如果需要包含内容
        if include_content:
            # 读取完整响应数据
            if response_exists:
                try:
                    with open(response_path, "r", encoding="utf-8") as f:
                        result["full_response"] = json.load(f)
                except Exception as e:
                    result["message"] += f"，但读取响应数据失败: {str(e)}"
                    # 如果读取JSON失败，尝试从TXT文件构建简单响应
                    if summary_exists:
                        try:
                            with open(summary_path, "r", encoding="utf-8") as f:
                                summary_content = f.read()
                                # 构建简单的响应对象
                                result["full_response"] = {
                                    "cid": cid,
                                    "summary": summary_content,
                                    "success": True,
                                    "message": "从摘要文件读取",
                                    "from_deepseek": True,
                                }
                        except Exception as e2:
                            result["message"] += f"，读取摘要文件也失败: {str(e2)}"
            # 如果没有响应数据文件，但有摘要文件
            elif summary_exists:
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary_content = f.read()
                        # 构建简单的响应对象
                        result["full_response"] = {
                            "cid": cid,
                            "summary": summary_content,
                            "success": True,
                            "message": "从摘要文件读取",
                            "from_deepseek": True,
                        }
                except Exception as e:
                    result["message"] += f"，但读取摘要内容失败: {str(e)}"

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查本地摘要文件时出错: {str(e)}")
