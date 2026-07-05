import asyncio
import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------
# 导入编译好的 LangGraph 健康管理应用
# ------------------------------------------------------------
_health_path = Path(__file__).parent / "graph.py"
_spec = importlib.util.spec_from_file_location("health_agent", _health_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {_health_path}")
_health_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_health_module)
health_app = _health_module.health_app

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    image_base64: str | None = None
    long_term_goal: str | None = None
    session_id: str = "default_session"
    thread_id: str | None = None
    user_profile: dict | None = None
    daily_log: dict | None = None


# ------------------------------------------------------------
# 默认用户档案与每日日志（前端未传时使用 Mock 数据）
# ------------------------------------------------------------
DEFAULT_USER_PROFILE = {
    "age": 21,
    "height": 171.6,
    "weight": 68.0,
    "bmr": 1578.0,
    "long_term_goal": "保持健康体重，提升体能",
}

DEFAULT_DAILY_LOG = {
    "sleep_hours": 7.0,
    "fatigue_level": 6,
    "recovery_score": 64.0,
    "exercise_target": "无",
}

# ------------------------------------------------------------
# 视觉 LLM 配置（智谱 GLM-4V-Flash）
# ------------------------------------------------------------
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")


async def analyze_image_with_llm(base64_str: str) -> str:
    """使用智谱 GLM-4V-Flash 识别图片中的食物"""
    if not ZHIPU_API_KEY:
        return "【图片识别系统：未配置 ZHIPU_API_KEY，无法进行视觉识别】"

    try:
        vision_llm = ChatOpenAI(
            model="glm-4.6v-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=ZHIPU_API_KEY,  # type: ignore
            temperature=0.1,
        )

        prompt = (
            "你是一个专业的数据与图像提取专家。请仔细观察图片内容并直接输出核心信息：\n"
            "1. 如果图片是食物，请准确识别菜品并列出主要食材。\n"
            "2. 如果图片是运动记录（如跑步配速表、心率图），请直接提取其中的核心数值（如总距离、总时长、平均配速、消耗热量等）。\n"
            "3. 如果是其他与健康或身体相关的内容，请列出关键指标。\n"
            "【严格要求】：直接输出提取到的核心数据或结论，绝对不要使用诸如‘这是一张...的图片’、‘图片显示了...’等毫无意义的开场白。"
        )

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
                },
            ]
        )

        response = await vision_llm.ainvoke([msg])

        result = str(response.content)

        if result:
            return f"【图片识别系统：{result.strip()}】"
        else:
            return "【图片识别系统：未能识别出图片中的食物，请尝试更换图片】"

    except Exception as e:
        print(f"视觉识别报错: {e}")
        return f"【图片识别系统：视觉识别暂时不可用（{type(e).__name__}），请手动描述图片内容】"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def generate():# ---------- 图片拦截：调用真实视觉大模型识别 ----------
        message_text = req.message or ""
        if req.image_base64:
            vision_text = await analyze_image_with_llm(req.image_base64)
            # 将识别结果拼接到用户原文最前面
            message_text = vision_text + "\n" + message_text if message_text else vision_text

        # 【✨ 核心修复 1：动态目标强制注入】
        # 为了防止 LLM 被长对话历史“绑架”，我们直接在每一次的用户真实消息末尾，
        # 附加上最高权重的系统提醒，强迫它立刻响应最新目标。
        if req.long_term_goal:
            message_text += f"\n\n[系统强化指令：用户当前的长期目标已更新为“{req.long_term_goal}”，请必须以此最新目标为准提供评估和建议！]"

        # 构造 HumanMessage
        user_msg = HumanMessage(content=message_text)

        # 优先使用前端传来的个性化用户档案，其次使用默认值
        if req.user_profile and isinstance(req.user_profile, dict):
            user_profile = dict(DEFAULT_USER_PROFILE)
            user_profile.update(req.user_profile)
        else:
            user_profile = dict(DEFAULT_USER_PROFILE)
        if req.long_term_goal:
            user_profile["long_term_goal"] = req.long_term_goal

        input_state = {
            "user_profile": user_profile,
            "daily_log": req.daily_log if req.daily_log else DEFAULT_DAILY_LOG,
            "chat_history": [user_msg],
        }

        # 优先使用前端传来的 thread_id 做会话隔离，否则回退 session_id
        effective_thread_id = req.thread_id or req.session_id
        config = {"configurable": {"thread_id": effective_thread_id}}

        async for chunk in health_app.astream(
            input_state, config, stream_mode="messages"
        ):
            message, _ = chunk
            if isinstance(message, AIMessage):
                content = message.content
                if isinstance(content, str) and content:
                    yield content
                    await asyncio.sleep(0.01)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
