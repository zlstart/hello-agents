from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RunnableConfig, interrupt, Command


# 定义状态结构
class State(dict):
    task: str
    approval: str


# 自动步骤：生成任务
def create_task(state: State):
    task = "审批预算申请（金额：5000元）"
    print(f"系统创建任务：{task}")
    state["task"] = task
    return state


# 人工步骤：等待人工输入
def human_approval(state: State):
    print("等待人工审批中...")
    # 触发人工中断（这里 LangGraph 会暂停执行，直到被 resume）
    approval = interrupt("请输入审批结果（批准 / 不批准）：")
    state["approval"] = approval
    return state


# 后续步骤：根据人工结果继续
def finalize(state: State):
    if state["approval"] == "批准":
        print(f"✅ 审批通过：{state['task']}")
    else:
        print(f"❌ 审批拒绝：{state['task']}")
    return state


# === 构建图 ===
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("create_task", create_task)
    graph_builder.add_node("human_approval", human_approval)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "create_task")
    graph_builder.add_edge("create_task", "human_approval")
    graph_builder.add_edge("human_approval", "finalize")
    graph_builder.add_edge("finalize", END)

    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


# === 运行示例 ===
graph = build_graph()

# 第一次运行 —— 执行到人工中断处
config = RunnableConfig(configurable={"thread_id": "12345"})
print("=== 第一次运行 ===")
state = graph.invoke({}, config=config)
print("第一次运行状态：", state)

# 模拟人工输入并恢复执行
print("\n=== 第二次运行（人工输入后继续） ===")
user_decision = input("是否批准？")
state = graph.invoke(Command(resume=user_decision), config=config)
print("第二次运行状态：", state)