from langgraph.graph import StateGraph, END
from .state import QueryGraphState
from app.core.logger import logger
# 导入所有节点函数
from .node.node_item_name_confirm import node_item_name_confirm
from .node.node_answer_output import node_answer_output
from .node.node_rerank import node_rerank
from .node.node_rrf import node_rrf
from .node.node_search_embedding import node_search_embedding
from .node.node_search_embedding_hyde import node_search_embedding_hyde
from .node.node_web_search_mcp import node_web_search_mcp

# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
builder.add_node("node_item_name_confirm", node_item_name_confirm) # 确认商品
builder.add_node("node_multi_search", lambda x: x)                 # 虚拟节点：多路搜索分叉点
builder.add_node("node_search_embedding", node_search_embedding)   # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
# builder.add_node("node_query_kg", node_query_kg)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
# builder.add_node("node_join", lambda x: {})                        # 虚拟节点：多路搜索合并点
builder.add_node("node_rrf", node_rrf)                             # 排序
builder.add_node("node_rerank", node_rerank)                       # 重排
builder.add_node("node_answer_output", node_answer_output)         # 生成


def have_answer(state):
    answer = state.get("answer","")
    if not answer:
        return False
    return True

builder.set_entry_point("node_item_name_confirm")
builder.add_conditional_edges(
    source="node_item_name_confirm",
    path=have_answer,
    path_map={
        True: "node_answer_output",
        False: "node_multi_search"
    }
)
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")

builder.add_edge("node_search_embedding","node_rrf")
builder.add_edge("node_search_embedding_hyde","node_rrf")
builder.add_edge("node_web_search_mcp","node_rrf")
builder.add_edge("node_rrf","node_rerank")
builder.add_edge("node_rerank","node_answer_output")
builder.set_finish_point("node_answer_output")

query_app = builder.compile()


def test_pdf_flow():
    print("\n==测试PDF文件处理流程==")
    # 模拟初始化状态
    initial_state = QueryGraphState(
        session_id="test_task_001",
        original_query="如何使用华为手机？",
        answer="有答案了",
        is_stream=True
    )

    # 运行图
    print("开始运行....")
    try:
        # 修正点：使用 .invoke() 方法
        result = query_app.invoke(initial_state)
        # 打印流程图
        # query_app.get_graph().print_ascii()
        print(query_app.get_graph().draw_mermaid())
        print("运行结束，最终的状态 keys:", result.keys())
    except Exception as e:
        print(f"运行报错：{e}")
        # 打印详细堆栈以便调试
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("开始测试")
    test_pdf_flow()