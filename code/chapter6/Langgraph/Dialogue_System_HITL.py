"""
智能搜索助手 - LangGraph + Tavily + Human-in-the-Loop（新版 API）
"""

import asyncio
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import interrupt, Command  # ✅ 新 HITL API

import os
from dotenv import load_dotenv
from tavily import TavilyClient

# 加载环境变量
load_dotenv()

# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]  # 包含所有消息的列表
    user_query: str        # 用户查询（真实用户输入）
    search_query: str      # 优化后的搜索查询（可被人修改确认）
    search_results: str    # Tavily搜索结果
    final_answer: str      # 最终答案
    step: str              # 当前步骤

# 初始化模型和Tavily客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def understand_query_node(state: SearchState) -> SearchState:
    """步骤1：理解用户查询并生成搜索关键词（内置 HITL 确认）"""

    # 获取最新用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    resp = llm.invoke([SystemMessage(content=understand_prompt)])
    resp_text = resp.content

    # 默认使用原始查询作为搜索词
    search_query = user_message
    if "搜索词：" in resp_text:
        search_query = resp_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in resp_text:
        search_query = resp_text.split("搜索关键词：")[1].strip()

    # ✅ Human-in-the-Loop：暂停等待人工确认/修改
    human_feedback = interrupt({
        "model_understanding": resp_text,
        "suggested_search_query": search_query,
        "prompt": "输入 yes 继续；或直接输入新的搜索关键词："
    })

    # 恢复后：如果人类输入不是 yes/空，则覆盖搜索词
    if isinstance(human_feedback, str) and human_feedback.strip() and human_feedback.lower() != "yes":
        search_query = human_feedback.strip()

    return {
        "user_query": user_message,  # ✅ 修正：存真实用户问题
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我理解您的需求：{resp_text}\n将使用搜索词：{search_query}")]
    }

def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2：使用Tavily API进行真实搜索"""
    query = state["search_query"]
    try:
        print(f"🔍 正在搜索: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5
        )

        # 处理搜索结果
        search_results = ""
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"

        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"

        if not search_results:
            search_results = "抱歉，没有找到相关信息。"

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }

    except Exception as e:
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }

def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3：基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }

    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

# 构建搜索工作流
def create_search_assistant():
    workflow = StateGraph(SearchState)
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

async def main():
    """主函数：运行智能搜索助手（支持 HITL 恢复）"""

    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return

    app = create_search_assistant()
    print("🔍 智能搜索助手启动！（已启用 Human-in-the-Loop）")
    print("(输入 'quit' 退出)\n")

    # 固定 thread_id，保证多轮/中断可恢复
    config = {"configurable": {"thread_id": "search-session-1"}}

    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        if not user_input:
            continue

        print("\n" + "="*60)

        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }

        # 可能会多次“中断→恢复”，直到到达 END
        pending_resume = None
        finished = False

        while not finished:
            # 首次正常跑；中断后用 Command(resume=...)
            if pending_resume is None:
                stream_iter = app.stream(initial_state, config=config, stream_mode="updates")
            else:
                stream_iter = app.stream(Command(resume=pending_resume), config=config, stream_mode="updates")
                pending_resume = None

            interrupted = False

            for event in stream_iter:
                # 1) 处理中断事件（payload 在 __interrupt__ 中）
                intr = event.get("__interrupt__")
                if intr:
                    # __interrupt__ 通常是一个包含 Interrupt 的序列；取出其 value
                    interrupt_obj = intr[0] if isinstance(intr, (list, tuple)) else intr
                    ivalue = getattr(interrupt_obj, "value", interrupt_obj)

                    print("\n🛑 Human-in-the-Loop：")
                    if isinstance(ivalue, dict):
                        print(f"🧠 模型理解为：{ivalue.get('model_understanding')}")
                        print(f"🔍 建议搜索关键词：{ivalue.get('suggested_search_query')}")
                        print(ivalue.get("prompt", "👉 输入 yes 继续；或输入新的搜索关键词："))
                    else:
                        print(ivalue)

                    pending_resume = input("> ").strip()
                    interrupted = True
                    break  # 跳出 for，下一轮用 Command(resume=...) 继续

                # 2) 正常节点增量事件：打印 AI 消息
                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue
                    msgs = node_output.get("messages") or []
                    if msgs:
                        latest_message = msgs[-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")

            if not interrupted:
                finished = True  # 未被中断，说明到达 END

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
