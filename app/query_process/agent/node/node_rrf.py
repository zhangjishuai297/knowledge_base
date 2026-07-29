import sys
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点
    
    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。
    
    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("开始进行rrf排序")
    #   1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks
    embedding_chunks = state.get("embedding_chunks",[])
    logger.info("获取的切片数量如下:")
    logger.info(f"embedding_chunks:{len(embedding_chunks)}")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks",[])
    logger.info(f"hyde_embedding_chunks:{len(hyde_embedding_chunks)}")
    #   2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    """
    {
		"id": 467592966412277916,
		"distance": 0.8236948847770691,
		"entity": {
			"chunk_id": 467592966412277916,
			"content": "## 用户指南\n",
			"item_name": "华为擎云W525"
		}
    }
    """
    # 3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）
    ranker_weights = (1.0, 1.0)
    merge_list = _rrf_func(embedding_chunks, hyde_embedding_chunks, ranker_weights)
    
    state["rrf_chunks"] = merge_list
    

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


def _rrf_func(embedding_chunks, hyde_embedding_chunks, ranker_weights=(1.0,1.0), k=60, max_results=10):
    # 定义以chunk_id为键的字典
    score_dict = {}
    # 定义存储具体chunk内容的字典,以chunk_id为键,最终和score_dict进行合并
    chunk_dict = {}
    for i, chunk in enumerate(embedding_chunks,start=1):
        # 取出chunk_id
        chunk_id = chunk["id"] if chunk.get("id") is not None else chunk["entity"]["chunk_id"]
        score_dict[chunk_id] = ranker_weights[0] /(i + k)
        chunk_dict.setdefault(chunk_id,chunk)

    # 遍历hyde_embedding_chunks,更新chunk_dict
    for i, chunk in enumerate(hyde_embedding_chunks,start=1):
        chunk_id = chunk["id"] if chunk.get("id") is not None else chunk["entity"]["chunk_id"]
        # 这里不需要做判断是否已存在,获取该值,没有则为0
        score_dict[chunk_id] = score_dict.get(chunk_id,0.0) + (ranker_weights[1] /(i + k))
        chunk_dict.setdefault(chunk_id,chunk)
        
    merge_list = []
    for chunk_id, score in score_dict.items():
        # 获取chunk_id对应的chunk
        chunk = chunk_dict.get(chunk_id)
        entity = chunk.get("entity") or chunk
        merge_list.append({
            "id": chunk_id,
            "score": score,
            "entity": entity
        })


    # 按照得出的分数进行排序
    merge_list.sort(key=lambda x:x["score"],reverse=True)
    logger.info(f"merge_list数量:{len(merge_list)}")
    # 判断是否超过最大结果数
    if len(merge_list) > max_results:
        merge_list = merge_list[:max_results]
    return merge_list