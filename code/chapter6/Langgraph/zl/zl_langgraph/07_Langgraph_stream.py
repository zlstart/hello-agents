from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict


# ===== 1. 定义状态 =====
class MyState(TypedDict):
    question: str
    thought: str
    answer: str


# ===== 3. 定义节点 =====
def think_node(state: MyState):
    """模拟思考过程"""
    print("🤔 思考中...")
    state["thought"] = "是怎么染色的呢？"
    return state


def answer_node(state: MyState):
    """生成最终答案"""
    print("\n✅ 回答完成！")
    state["answer"] = "是用彩笔染色的。"
    return state


# ===== 4. 构建图 =====
def build_graph():
    graph_builder = StateGraph(MyState)
    graph_builder.add_node("思考节点", think_node)
    graph_builder.add_node("回答节点", answer_node)

    graph_builder.add_edge(START, "思考节点")
    graph_builder.add_edge("思考节点", "回答节点")
    graph_builder.add_edge("回答节点", END)

    return graph_builder.compile()


# ===== 5. 测试运行 =====
# Langgraph的输出可以设置成节点流式输出。
if __name__ == "__main__":
    graph = build_graph()
    for chunk in graph.stream({"question": "为什么天空是蓝色的？"}):
        print(chunk)