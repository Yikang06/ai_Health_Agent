from typing import TypedDict, Annotated, cast, Any
import operator


class UserProfile(TypedDict):
    age: int
    height: float                    # 单位：cm
    weight: float                    # 单位：kg
    bmr: float                       # 基础代谢率（千卡/天）
    long_term_goal: str              # 长期健康目标


class DailyLog(TypedDict):
    sleep_hours: float
    fatigue_level: int
    recovery_score: float
    exercise_target: str


class FoodItem(TypedDict):
    name: str                        # 食物名称
    calories: float                  # 卡路里（千卡）
    protein: float                   # 蛋白质（克）
    carbs: float                     # 碳水（克）
    fat: float                       # 脂肪（克）


class ChatMessage(TypedDict):
    role: str                        # "user" 或 "assistant"
    content: str


class HealthState(TypedDict):
    user_profile: UserProfile
    daily_log: DailyLog
    diet_intake: Annotated[list[FoodItem], operator.add]
    chat_history: Annotated[list, operator.add]          # 存放 LangChain 消息对象
    current_analysis: str


from langchain_core.tools import tool


@tool
def calculate_nutrition(food_description: str) -> dict:
    """
    根据用户输入的食物描述，估算该食物的营养素含量。

    适用场景：
    - 用户在对话中描述自己吃了什么（如 "一碗米饭加炒肉"），
      需要获取该食物的卡路里、蛋白质、碳水、脂肪等数据时调用。

    参数:
        food_description (str): 用户对食物的自然语言描述，例如 "一碗米饭加青椒炒肉"。

    返回:
        dict: 包含以下键的字典:
            - name (str): 食物名称
            - calories (float): 估算热量（千卡）
            - protein (float): 估算蛋白质（克）
            - carbs (float): 估算碳水（克）
            - fat (float): 估算脂肪（克）
    """
    # Mock 数据：根据关键词返回预设营养素
    description = food_description.lower()
    if "米饭" in description or "饭" in description:
        if "炒" in description or "肉" in description:
            return {
                "name": "米饭加炒肉",
                "calories": 650.0,
                "protein": 25.0,
                "carbs": 80.0,
                "fat": 22.0,
            }
        return {
            "name": "白米饭",
            "calories": 260.0,
            "protein": 5.0,
            "carbs": 57.0,
            "fat": 0.5,
        }
    if "面" in description:
        return {
            "name": "面条",
            "calories": 350.0,
            "protein": 12.0,
            "carbs": 60.0,
            "fat": 6.0,
        }
    if "沙拉" in description or "蔬菜" in description:
        return {
            "name": "蔬菜沙拉",
            "calories": 120.0,
            "protein": 3.0,
            "carbs": 10.0,
            "fat": 7.0,
        }
    if "鸡蛋" in description:
        return {
            "name": "鸡蛋",
            "calories": 155.0,
            "protein": 13.0,
            "carbs": 1.0,
            "fat": 11.0,
        }
    # 兜底：返回通用估算
    return {
        "name": food_description,
        "calories": 400.0,
        "protein": 15.0,
        "carbs": 50.0,
        "fat": 15.0,
    }

from rag_builder import query_health_knowledge

@tool
def search_health_guidelines(query: str) -> str:
    """
    查询专业的健康与营养学文献知识库。
    
    适用场景：
    - 用户询问一般的健康常识、科学减脂/增肌原理。
    - 用户询问“饮食指南”、“每天应该喝多少水”、“如何搭配饮食”等需要权威依据的问题。
    
    参数:
        query (str): 用户的查询问题提取出的关键词或短语。
        
    返回:
        str: 检索到的相关权威文献片段。
    """
    return query_health_knowledge(query)

@tool
def evaluate_recovery_score(sleep_hours: float, fatigue_level: int) -> str:
    """
    根据用户昨晚的睡眠时长和今日主观疲劳度（1-10分），
    计算身体恢复分数并返回简要评估。

    适用场景：
    - 用户提供了睡眠数据和疲劳感受时调用，用于评估当前身体恢复状态。
    - Agent 需要根据恢复情况给出运动或休息建议前调用。

    参数:
        sleep_hours (float): 昨晚睡眠时长，单位小时，如 7.5。
        fatigue_level (int): 今日主观疲劳度评分，范围 1（精力充沛）到 10（极度疲惫）。

    返回:
        str: 一段简短的恢复状态评估，包含恢复分数和文字说明。
    """
    # 疲劳度越高说明恢复越差，睡眠越长恢复越好
    sleep_score = min(sleep_hours / 8.0, 1.0) * 100   # 以8小时为满分基准
    fatigue_score = (11 - fatigue_level) / 10.0 * 100  # 疲劳度越低得分越高

    recovery_score = round(0.6 * sleep_score + 0.4 * fatigue_score, 1)

    if recovery_score >= 80:
        level = "优秀"
        advice = "身体状态良好，适合进行中高强度运动。"
    elif recovery_score >= 60:
        level = "良好"
        advice = "基本恢复，建议进行中等强度运动，注意补充营养。"
    elif recovery_score >= 40:
        level = "一般"
        advice = "恢复不足，建议以轻度活动（如散步、拉伸）为主，保证充足休息。"
    else:
        level = "较差"
        advice = "身体明显未恢复，建议今天暂停高强度训练，优先补觉和补充营养。"

    return (
        f"恢复评分：{recovery_score}/100（{level}）\n"
        f"睡眠时长 {sleep_hours}h，疲劳度 {fatigue_level}/10。\n"
        f"建议：{advice}"
    )


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os

# ------------------------------------------------------------
# 1. 构建 StateGraph
# ------------------------------------------------------------
graph = StateGraph(HealthState)

# ------------------------------------------------------------
# 2. 初始化 LLM（使用 DeepSeek API；未配置密钥时回退到 Mock）
# ------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

if DEEPSEEK_API_KEY:
    llm = ChatOpenAI(
        model_name="deepseek-v4-pro",                    # DeepSeek-V4-pro 对话模型      # pyright: ignore[reportCallIssue]
        openai_api_base="https://api.deepseek.com/v1",  # DeepSeek 兼容 OpenAI 格式  # pyright: ignore[reportCallIssue]
        openai_api_key=DEEPSEEK_API_KEY,     # pyright: ignore[reportCallIssue]
        temperature=0.7,
    )
else:
    # 未配置 API Key：使用 FakeListChatModel 模拟大模型回复，方便本地调试
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    print("[WARNING] 未设置环境变量 DEEPSEEK_API_KEY，使用 Mock 模型运行。")
    llm = FakeListChatModel(
        responses=cast(Any, [
            # 第一轮：调用 evaluate_recovery_score 工具
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "evaluate_recovery_score",
                        "args": {"sleep_hours": 7.0, "fatigue_level": 6},
                        "id": "mock_call_1",
                    }
                ],
            ),
            # 第二轮（工具返回后）：调用 calculate_nutrition
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_nutrition",
                        "args": {"food_description": "米饭加炒肉"},
                        "id": "mock_call_2",
                    }
                ],
            ),
            # 第三轮（所有工具返回后）：最终健康建议
            AIMessage(
                content=(
                    "根据今天的综合数据分析，以下是你今日的健康建议：\n\n"
                    "🥗 **饮食方面**：你今天摄入约 650 千卡，蛋白质摄入 25g，"
                    "建议晚餐增加蔬菜和优质蛋白（如鸡胸肉、鱼类）。\n\n"
                    "💤 **恢复状态**：昨晚睡眠 7 小时，疲劳度评分为 6/10，"
                    "恢复评分约 65/100（良好）。建议今晚保证 7-8 小时睡眠。\n\n"
                    "🏃 **运动建议**：今天适合中等强度运动，如慢跑 30 分钟或快走 45 分钟。\n\n"
                    "长期坚持良好的饮食和作息习惯，你的健康目标一定能达成！💪"
                ),
            ),
        ]
    ))

# ------------------------------------------------------------
# 3. 工具列表 & 绑定到 LLM
# ------------------------------------------------------------
tools = [calculate_nutrition, evaluate_recovery_score, search_health_guidelines]
llm_with_tools = llm.bind_tools(tools)


# ------------------------------------------------------------
# 3. agent 节点
#    让模型根据用户输入 + 当前状态决定：调用工具 或 直接回答
# ------------------------------------------------------------
def agent_node(state: HealthState) -> dict:
    daily = state.get('daily_log', {})
    daily_str = (
        f"睡眠时长：{daily.get('sleep_hours', '未知')}小时，"
        f"疲劳度：{daily.get('fatigue_level', '未知')}/10，"
        f"系统评定恢复分：{daily.get('recovery_score', '未知')}/100，"
        f"今日运动目标：【{daily.get('exercise_target', '无')}】。"
    )

    system_prompt = (
        "你是一个极其专业的健康与恢复管理助手（Agent）。\n"
        "你可以使用以下工具：\n"
        "1. calculate_nutrition — 估算食物的卡路里和营养素。\n"
        "2. evaluate_recovery_score — 根据睡眠和疲劳度评估身体恢复状态。\n"
        "3. search_health_guidelines — 检索权威的《中国居民膳食指南》和健康知识库。\n"
        "【工作流要求】：\n"
        "当用户描述饮食或询问恢复状态时，请主动调用前两个工具获取数据。\n"
        "当用户询问科普知识、生活建议或原理时，必须调用 search_health_guidelines 寻找文献支撑。\n"
        "【⚠️ 绝对安全边界（防越狱）】：\n"
        "你的服务领域仅限：运动科学、饮食营养、健康作息和身体恢复。\n"
        "如果用户询问与此无关的话题（如：编写计算机代码、金融股票投资、时政新闻、数学解题等），\n"
        "你必须礼貌但坚决地拒绝回答，并引导用户回到健康话题。绝对不能输出任何非健康领域的专业内容或代码片段！\n"
        f"【🚨 当前用户档案与目标】（最高优先级）：\n"
        f"用户身体数据：年龄 {state.get('user_profile', {}).get('age')}，身高 {state.get('user_profile', {}).get('height')}cm，体重 {state.get('user_profile', {}).get('weight')}kg，BMR {state.get('user_profile', {}).get('bmr')}。\n"
        f"🎯 用户当前长期目标：【{state.get('user_profile', {}).get('long_term_goal')}】\n"
        "请你在每一次提供饮食推荐、运动建议或回答问题时，**必须强制结合上述长期目标**！即便用户的问题看似与目标无关，你也必须在回答中体现出对该目标的兼顾。\n"
        f"【📝 今日动态日志】：\n"
        f"{daily_str}\n"
        "注意：用户的恢复评分和运动目标已由系统前置计算，生成报告或给建议时，请直接引用上述数据，不需要再调用评估工具估算。"
    )

    # ========== 滑动窗口：只保留最后 10 条消息，防止长对话 Token 溢出 ==========
    chat_history = state["chat_history"]
    MAX_HISTORY = 50
    recent_history = chat_history[-MAX_HISTORY:] if len(chat_history) > MAX_HISTORY else list(chat_history)

    # ========== 安全拦截器：确保第一条消息以 HumanMessage 开头 ==========
    # 国内大模型 API（DeepSeek 等）强制要求多轮对话上下文必须以人类消息开头
    while recent_history and not isinstance(recent_history[0], HumanMessage):
        recent_history = recent_history[1:]

    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + recent_history
    )

    # 追加到 chat_history（SystemMessage 不入状态，LangGraph 底层自动追加）
    return {"chat_history": [response]}


# ------------------------------------------------------------
# 4. tools 节点 —— 标准 ToolNode
# ------------------------------------------------------------
tool_node = ToolNode(tools, messages_key="chat_history")


# ------------------------------------------------------------
# 5. state_update 节点 —— 将工具返回的食物数据写入 diet_intake
# ------------------------------------------------------------
import json
from langchain_core.messages import ToolMessage


def state_update_node(state: HealthState) -> dict:
    """
    在 tools 节点执行完后运行，检查最后一条 ToolMessage：
    - 如果来自 calculate_nutrition，解析返回的食物数据追加到 diet_intake
    - 其他工具（如 evaluate_recovery_score）不做状态写入
    """
    result = {}  # 空字典 = 不更新任何字段

    for msg in reversed(state["chat_history"]):
        if not isinstance(msg, ToolMessage):
            break  # 只处理紧邻的 ToolMessage

        if msg.name == "calculate_nutrition":
            if not isinstance(msg.content, str):
                continue
            try:
                food_data = json.loads(msg.content)
                food_item = {
                    "name": food_data.get("name", "未知食物"),
                    "calories": float(food_data.get("calories", 0)),
                    "protein": float(food_data.get("protein", 0)),
                    "carbs": float(food_data.get("carbs", 0)),
                    "fat": float(food_data.get("fat", 0)),
                }
                result["diet_intake"] = [food_item]
            except (json.JSONDecodeError, TypeError):
                pass  # 解析失败则跳过，不写入

    return result


# ------------------------------------------------------------
# 6. 条件边 —— 实现 ReAct 循环
# ------------------------------------------------------------
def should_continue(state: HealthState) -> str:
    """
    检查最后一条消息是否包含 tool_calls：
    - 有 tool_calls → 路由到 "tools" 节点
    - 没有        → 结束
    """
    last_message = state["chat_history"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ------------------------------------------------------------
# 7. 组装 Graph
# ------------------------------------------------------------
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("state_update", state_update_node)

graph.add_edge(START, "agent")                            # 入口 → agent
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "state_update")                   # 工具执行完 → 状态更新
graph.add_edge("state_update", "agent")                   # 状态更新完 → 回到 agent

# ------------------------------------------------------------
# 8. 编译（带 MemorySaver 持久化）
# ------------------------------------------------------------
memory = MemorySaver()
health_app = graph.compile(checkpointer=memory)
