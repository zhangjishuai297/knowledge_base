from email import message
import time
import sys
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.node.node_item_name_confirm import node_item_name_confirm
from app.utils.task_utils import  add_done_task,add_running_task
from app.core.logger import logger
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 用重写后的query进行向量匹配
    item_names = state["item_names"]
    if not item_names:
        logger.warning("item_names is empty,跳过")
        return {"hyde_embedding_chunks":[]}
    rewritten_query = state.get("rewritten_query")
    if rewritten_query:
        query = rewritten_query
    else:
        query = state.get("original_query", "")
    
    # 获取大模型,先从大模型获取答案,再根据大模型给的答案检索
    llm_client = get_llm_client()
    prompt = load_prompt("hyde_prompt",rewritten_query=query)
    messages = [{"role": "user", "content": prompt}]
    llm_res = llm_client.invoke(messages)
    content = llm_res.content
    final_content = rewritten_query + "\n" + content
    logger.info(f"最终需要生成向量的文本:{final_content}")
    
    
        
    # 获取Milvus客户端
    milvus_client = get_milvus_client()
    #调用嵌入模型,进行向量化
    res = generate_embeddings([final_content])
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
    hyde_embedding_chunks = []
    if hybrid_result:
        logger.info(f"检索到有结果：{len(hybrid_result)}条")
        hyde_embedding_chunks = hybrid_result[0]
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    return {"hyde_embedding_chunks":hyde_embedding_chunks}

if __name__ == "__main__":
    query_state = {"original_query": "华为擎云W525怎么使用","session_id":"user_dong","is_stream":False}
    new_state = node_item_name_confirm(query_state)
    
    # new_state = node_search_embedding(new_state)
    new_state = node_search_embedding_hyde(new_state)
    logger.info(f"主体确认结果: {new_state}")