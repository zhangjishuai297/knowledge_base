from turtle import st
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.node.node_item_name_confirm import node_item_name_confirm
from app.utils.task_utils import add_running_task,add_done_task
from app.query_process.agent.state import QueryGraphState
import sys
from app.core.logger import logger
def node_search_embedding(state: QueryGraphState):
    
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 用重写后的query进行向量匹配
    item_names = state["item_names"]
    if not item_names:
        logger.warning("item_names is empty,跳过")
        return {"embedding_chunks":[]}
    rewritten_query = state.get("rewritten_query")
    if rewritten_query:
        query = rewritten_query
    else:
        query = state.get("original_query", "")
        
    # 获取Milvus客户端
    milvus_client = get_milvus_client()
    #调用嵌入模型,进行向量化
    res = generate_embeddings([query])
    dense_vectors = res.get("dense", [])
    sparse_vectors = res.get("sparse", [])
    dense_vector = dense_vectors[0] if dense_vectors else None
    sparse_vector = sparse_vectors[0] if sparse_vectors else None
    
    # 创建混合搜索请求对象
    item_names_expr = ",".join(f"\"{i_name}\"" for i_name in item_names)
    expr = f"item_name in [{item_names_expr}]"
    reqs = create_hybrid_search_requests(dense_vector=dense_vector,
                                         sparse_vector=sparse_vector,
                                         expr=expr,
                                         limit=10)
    # 进行混合搜索
    hybrid_result = hybrid_search(client=milvus_client,
                    collection_name=milvus_config.chunks_collection,
                    reqs=reqs,
                    ranker_weights=(0.9,0.1),
                    norm_score=True,
                    output_fields=["chunk_id", "content", "item_name"])
    embedding_chunks = []
    if hybrid_result:
        logger.info(f"检索到有结果：{len(hybrid_result)}条")
        embedding_chunks = hybrid_result[0]
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    return {"embedding_chunks":embedding_chunks}

if __name__ == "__main__":
    query_state = {"original_query": "华为擎云W525怎么使用","session_id":"user_dong","is_stream":False}
    new_state = node_item_name_confirm(query_state)
    
    new_state = node_search_embedding(new_state)
    logger.info(f"主体确认结果: {new_state}")