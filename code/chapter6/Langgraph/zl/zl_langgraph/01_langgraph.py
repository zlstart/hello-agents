from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable


# ========== 1) 配置 LLM ==========
# 这里保留你当前的 DeepSeek 配置，便于直接运行
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-ee0edf0ac2ee455b8e00192d5ee049c1"
)

# ========== 2) 定义状态 ==========
class AgentState(TypedDict):
    # 输入
    input: str
    # 是否是翻译
    is_translate_or_not: bool
    # 要翻译的句子
    translate_sentence:str
    # 翻译结果
    translate_result: str
    # QA结果
    qa_result: str
    # 输出
    output: str



def identify_intent_node(state: AgentState) -> AgentState:
    """判断用户输入是翻译文章还是普通回答"""
    input = state['input']
    messages = [
        SystemMessage(content=(
            "你是一个意图分类助手，用来判断用户输入是'翻译任务'还是'普通回答任务'。\n"
            "翻译任务的特征：用户明确要求翻译文章、句子、单词，或要求将中文翻译成英文/英文翻译成中文。\n"
            "普通回答任务的特征：用户是直接提问、咨询信息、讨论话题，而不是要求翻译。\n"
            "请你只回答：翻译 或 普通回答，不要补充其他内容。"
        )),
        HumanMessage(content=input)
    ]

    response = llm.invoke(messages).content.strip()

    if "翻译" in response:
        state['is_translate_or_not'] = True
    else:
        state['is_translate_or_not'] = False

    return state


def extract_translate_sentence_node(state: AgentState) -> AgentState:
    """提取翻译句子"""
    input = state['input']
    prompt = (
        "你是翻译助手。请先自动识别语言，然后提取待翻译的句子。"
        "请将待翻译的句子提取出来，并输出。"
        "请不要输出其他内容。\n"
        f"待翻译内容：{input}"
    )
    result = llm.invoke([HumanMessage(content=prompt)]).content
    state['translate_sentence'] = result
    return state

def translate_node(state: AgentState) -> AgentState:
    """翻译节点"""
    translate_sentence = state["translate_sentence"]
    prompt = (
        "你是翻译助手。请先自动识别语言，然后在中文与英文之间互译。"
        "只输出译文本身，不要额外解释。\n"
        f"待翻译内容：{translate_sentence}"
    )
    result = llm.invoke([HumanMessage(content=prompt)]).content
    state['translate_result'] = result
    state['output'] = result
    # 按你的状态定义回写
    return state


def qa_node(state: AgentState) -> AgentState:
    """QA节点"""
    qa_sentence = state["input"]
    prompt = (
        "你是一个问答助手，请回答问题。"
        "请使用中文回答。\n"
        f"问题：{qa_sentence}"
    )
    result = llm.invoke([HumanMessage(content=prompt)]).content
    state['qa_result'] = result
    state['output'] = result
    # 按你的状态定义回写
    return state




# 构建langgraph
def build_langgraph():
    graph_builder = StateGraph(AgentState)
    # 添加节点
    graph_builder.add_node("意图识别节点", identify_intent_node)
    graph_builder.add_node("抽取翻译内容节点", extract_translate_sentence_node)
    graph_builder.add_node("翻译节点", translate_node)
    graph_builder.add_node("问答节点", qa_node)
    # 添加边
    graph_builder.add_edge(START, "意图识别节点")

    def intent_detector_agent_control(state: AgentState):
        # 根据问题是否为中医问题，决定下一步的代理节点
        if state['is_translate_or_not']:
            return "抽取翻译内容节点"
        else:
            return "问答节点"

    graph_builder.add_conditional_edges("意图识别节点", intent_detector_agent_control)
    graph_builder.add_edge("抽取翻译内容节点", "翻译节点")
    graph_builder.add_edge("翻译节点", END)
    graph_builder.add_edge("问答节点", END)
    graph = graph_builder.compile()
    return graph


from langgraph.graph.state import CompiledStateGraph

def output_pic_graph(graph: CompiledStateGraph, filename: str = "graph.jpg"):
    try:
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(filename, 'wb') as f:
            f.write(mermaid_code)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    graph = build_langgraph()
    # 测试翻译任务
    state = graph.invoke({"input": "请将这句话翻译成英文：你好，世界！"})
    print(state['output'])
    # 测试普通问答任务
    state = graph.invoke({"input": "解释下什么是大模型"})
    print(state['output'])
    # 输出langgraph图片,没啥用，条件边展示不出来
    output_pic_graph(graph)