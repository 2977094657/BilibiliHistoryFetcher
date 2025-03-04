"""
DeepSeek API 路由
提供与DeepSeek大语言模型交互的API接口
"""

import json
import os
from datetime import datetime
from typing import Optional, List

import aiohttp
import requests
import yaml
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# 创建API路由
router = APIRouter()

# 定义请求和响应模型
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色，可以是'user'、'assistant'或'system'")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="聊天消息列表")
    model: Optional[str] = Field(None, description="模型名称，例如'deepseek-chat'")
    temperature: Optional[float] = Field(None, description="温度参数，控制生成文本的随机性")
    max_tokens: Optional[int] = Field(None, description="最大生成的token数量")
    top_p: Optional[float] = Field(None, description="核采样参数")
    stream: Optional[bool] = Field(False, description="是否使用流式输出")
    json_mode: Optional[bool] = Field(False, description="是否启用JSON输出模式")

class StreamResponse(BaseModel):
    content: str = Field(..., description="当前生成的内容片段")
    finish_reason: Optional[str] = Field(None, description="完成原因")

class TokenDetails(BaseModel):
    cached_tokens: Optional[int] = Field(0, description="缓存的token数量")

class UsageInfo(BaseModel):
    prompt_tokens: int = Field(..., description="提示tokens数量")
    completion_tokens: int = Field(..., description="完成tokens数量")
    total_tokens: int = Field(..., description="总tokens数量")
    prompt_tokens_details: Optional[TokenDetails] = Field(None, description="提示tokens详情")

class ChatResponse(BaseModel):
    content: str = Field(..., description="生成的内容")
    model: str = Field(..., description="使用的模型")
    usage: UsageInfo = Field(..., description="Token使用情况")
    finish_reason: Optional[str] = Field(None, description="完成原因")

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "deepseek"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# 加载YAML配置文件
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return {}

# 获取配置
config = load_config()
deepseek_config = config.get('deepseek', {})

# 设置API密钥（优先使用环境变量，其次使用配置文件）
API_KEY = os.environ.get("DEEPSEEK_API_KEY", deepseek_config.get('api_key', ''))
API_BASE = deepseek_config.get('api_base', 'https://api.deepseek.com/v1')
DEFAULT_MODEL = deepseek_config.get('default_model', 'deepseek-chat')

# 辅助函数，用于记录API调用日志
async def log_api_call(model: str, prompt_tokens: int, completion_tokens: int):
    """记录API调用日志，可以扩展为保存到数据库或发送到监控系统"""
    usage_info = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
    
    # 未来可添加写入数据库或发送到监控系统的代码
    print(f"DeepSeek API调用: {usage_info}")

@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    与DeepSeek API进行聊天交互
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API密钥未配置，请在config/config.yaml中设置deepseek.api_key或设置DEEPSEEK_API_KEY环境变量")
    
    # 准备API调用
    url = f"{API_BASE}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 从配置中获取默认设置
    default_settings = deepseek_config.get('default_settings', {})
    
    # 请求数据
    data = {
        "model": request.model or DEFAULT_MODEL,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature if request.temperature is not None else default_settings.get("temperature", 1.0),
        "max_tokens": request.max_tokens if request.max_tokens is not None else default_settings.get("max_tokens", 1000),
    }
    
    # 添加可选参数
    if request.top_p is not None:
        data["top_p"] = request.top_p
    
    # 如果启用JSON模式
    if request.json_mode:
        data["response_format"] = {"type": "json_object"}
    
    # 如果启用流式输出，则抛出异常（应该使用stream端点）
    if request.stream:
        raise HTTPException(status_code=400, detail="流式输出请使用 /deepseek/stream 端点")
    
    try:
        # 发送请求
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 抛出HTTP错误，如果有的话
        
        # 获取响应
        result = response.json()
        
        # 提取内容
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish_reason = result.get("choices", [{}])[0].get("finish_reason")
        
        # 获取Token使用量
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        prompt_tokens_details = usage.get("prompt_tokens_details", {"cached_tokens": 0})
        
        # 添加后台任务记录API调用
        background_tasks.add_task(
            log_api_call,
            request.model or DEFAULT_MODEL,
            prompt_tokens,
            completion_tokens
        )
        
        return {
            "content": content,
            "model": request.model or DEFAULT_MODEL,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_tokens_details": prompt_tokens_details
            },
            "finish_reason": finish_reason
        }
    except requests.exceptions.RequestException as e:
        error_message = f"API调用出错: {str(e)}"
        if hasattr(e, 'response') and e.response:
            error_message += f"\n错误详情: {e.response.text}"
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/stream")
async def stream_completion(request: ChatRequest):
    """
    与DeepSeek API进行流式交互
    
    注意：此函数返回的是一个流式响应，不同于普通的JSON响应
    """
    # 流式输出必须为True
    if not request.stream:
        request.stream = True
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API密钥未配置，请在config/config.yaml中设置deepseek.api_key或设置DEEPSEEK_API_KEY环境变量")
    
    # 准备API调用
    url = f"{API_BASE}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 从配置中获取默认设置
    default_settings = deepseek_config.get('default_settings', {})
    
    # 请求数据
    data = {
        "model": request.model or DEFAULT_MODEL,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature if request.temperature is not None else default_settings.get("temperature", 1.0),
        "max_tokens": request.max_tokens if request.max_tokens is not None else default_settings.get("max_tokens", 1000),
        "stream": True  # 启用流式输出
    }
    
    # 添加可选参数
    if request.top_p is not None:
        data["top_p"] = request.top_p
    
    # 如果启用JSON模式
    if request.json_mode:
        data["response_format"] = {"type": "json_object"}
    
    try:
        # 创建一个异步生成器函数处理流式响应
        async def generate():
            # 使用同步请求获取流式响应
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                response.raise_for_status()  # 抛出HTTP错误，如果有的话
                
                # 返回SSE格式的流式响应
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        # 处理SSE格式数据
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 跳过'data: '
                            if data_str == '[DONE]':
                                yield f"data: {json.dumps({'content': '', 'finish_reason': 'stop'})}\n\n"
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                finish_reason = data.get('choices', [{}])[0].get('finish_reason')
                                yield f"data: {json.dumps({'content': content, 'finish_reason': finish_reason})}\n\n"
                            except json.JSONDecodeError:
                                yield f"data: {json.dumps({'content': '[解析错误]', 'finish_reason': None})}\n\n"
        
        # 返回流式响应
        from fastapi.responses import StreamingResponse
        return StreamingResponse(generate(), media_type="text/event-stream")
    except requests.exceptions.RequestException as e:
        error_message = f"API调用出错: {str(e)}"
        if hasattr(e, 'response') and e.response:
            error_message += f"\n错误详情: {e.response.text}"
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/models", response_model=ModelList, summary="列出可用的DeepSeek模型")
async def list_models():
    """
    列出可用的DeepSeek模型列表，并提供相关模型的基本信息
    
    Returns:
        包含模型列表的响应对象，每个模型包含id、类型和所有者信息
    """
    try:
        api_key = config.get('deepseek', {}).get('api_key')
        if not api_key:
            raise HTTPException(status_code=401, detail="未配置DeepSeek API密钥")
            
        api_base = config.get('deepseek', {}).get('api_base', 'https://api.deepseek.com/v1')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"DeepSeek API请求失败: {error_msg}"
                    )
                    
                data = await response.json()
                return ModelList(
                    object="list",
                    data=[
                        ModelInfo(
                            id=model["id"],
                            object=model.get("object", "model"),
                            owned_by=model.get("owned_by", "deepseek")
                        )
                        for model in data.get("data", [])
                    ]
                )
                
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"请求DeepSeek API时发生错误: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表时发生错误: {str(e)}"
        ) 