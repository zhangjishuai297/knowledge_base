import sys
from app.core.logger import logger
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task, add_done_task

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5

def node_rerank(state):
    """
    Rerank节点
    对检索到的文档进行重新排序，提高相关性
    # 阶段一：合并文档
    # 阶段二：对文档进行重排序
    # 阶段三：动态 TopK
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.add("Rerank节点开始执行...")
    # 获取相关数据
    rrf_chunks = state.get("rrf_chunks",[])
    web_search_docs = state.get("web_search_docs",[])
    logger.info(f"开始进行rrf排序, rrf_chunks={len(rrf_chunks)}, web_search_docs={len(web_search_docs)}")
    if not rrf_chunks or not web_search_docs:
        return state
   
    # 合并文档
    merged_docs = step1_merge_docs(rrf_chunks, web_search_docs)
    # 获取重写问题,或者使用原始问题
    rewritten_query = state.get("rewritten_query",state.get("original_query"))
    if not rewritten_query:
        logger.error("无重写问题,无法进行rerank")
        return state
    
    # 使用重排序模型进行排序
    logger.info(f"开始进行rerank排序, rewritten_query={rewritten_query}")
    reranked_docs = step2_sort_docs(merged_docs,rewritten_query)
    # 阶段三：动态 TopK
    logger.info(f"开始进行动态TopK排序, reranked_docs={len(reranked_docs)}")
    final_docs = step3_dynamic_topk(reranked_docs)
    
    

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs":final_docs}

def step3_dynamic_topk(reranked_docs):
    """
    算法:
        防断崖算法,# 断崖阈值（相对）: (上一个分数-下一个分数)/上一个分数的绝对值大于阈值，则截断
                    RERANK_GAP_RATIO: 0.25
                    # 断崖阈值（相对）: (上一个分数-下一个分数)/上一个分数的绝对值大雨阈值，则截断
                    RERANK_GAP_RATIO: float = 0.25
                    # 断崖阈值（绝对）:2个相邻x的距离大于阈值,截断
                    RERANK_GAP_ABS: float = 0.5
        其他参数:
                # 动态 TopK 硬上限：最多取前 N 条（<=10）
                RERANK_MAX_TOPK: int = 10
                # 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
                RERANK_MIN_TOPK: int = 1
    """
    # 列表的长度
    doc_len = len(reranked_docs)
    # 计算需要循环的列表长度,和RERANK_MAX_TOPK的大小
    max_topk = min(doc_len, RERANK_MAX_TOPK)
    # 计算循环的开始位置,RERANK_MIN_TOPK,至少要保留RERANK_MIN_TOPK个
    min_topk = max(0, RERANK_MIN_TOPK)
    
    # 记录最后需要截断的索引
    topk = max_topk
    # 如果循环的列表长度小于等于RERANK_MIN_TOPK,则所有值都需要,不需要进行算法取值
    if max_topk > RERANK_MIN_TOPK:
        
        for index in range(min_topk-1, max_topk-1):
            # 根据索引当前分数和下一个分数
            s1_doc = reranked_docs[index]
            s2_doc = reranked_docs[index+1]
            s1_score = s1_doc.get("score")
            s2_score = s2_doc.get("score")
            # 计算差值
            gap = abs(s1_score - s2_score)
            # 计算差值比
            gap_ratio = gap / (abs(s1_score) +  +1e-6)
            
            # 做判断
            if gap >= RERANK_GAP_ABS or gap_ratio >= RERANK_GAP_RATIO:
                # 断崖阈值（相对）: (上一个分数-下一个分数)/上一个分数的绝对值大于阈值，则截断
                # 断崖阈值（绝对）:2个相邻x的距离大于阈值,截断
                # 记录截断索引
                topk = index + 1
                logger.info(f"断崖截断  index={index}, score: {s1_score:.4f} → {s2_score:.4f}")

                break
    logger.info(f"TopK 完成: 保留 {topk if topk < max_topk else max_topk} 条")

    return reranked_docs[0:topk]
            
            
            
            
            
    
def step2_sort_docs(merged_docs, rewritten_query):
    # 获取模型
    reranker_model = get_reranker_model()
    """
    compute_score方法
        param:
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]]
            数据格式Union代表2选1,
            格式①：单个句子对 (str, str)
            格式②：多个句子对组成的列表 [(str,str), ...]
            方法要求(问题,待打分的片段text)
        return:
            返回一个列表，列表元素为分数，对应输入的句子对,顺序和sentence_pairs一致
            由于sentence_pairs列表通过merged_docs构建,顺序和merged_docs一致

    """
    # 构建sentence_pairs
    sentence_pairs = [(rewritten_query, doc.get("text")) for doc in merged_docs]
    # normalize=True 结果为归一化后的分数,结果为正数,越接近1,越相似
    # 进行批处理
    bacth_size = 3
    score_list = []
    if len(sentence_pairs) > bacth_size:
        logger.info(f"进行批处理, bacth_size={bacth_size}")
        for i in range(0, len(sentence_pairs), bacth_size):
            score_list.extend(reranker_model.compute_score(sentence_pairs = sentence_pairs[i:i+bacth_size],normalize=True))
    else:
        score_list = reranker_model.compute_score(sentence_pairs = sentence_pairs,normalize=True)
    

    # 构建新列表
    new_doc_list = []
    
    for score, doc  in zip(score_list, merged_docs):
        # 拷贝doc,添加score
        new_doc = dict(doc)
        new_doc["score"] = score
        new_doc_list.append(new_doc)
    
    # 对新构建的列表进行排序
    new_doc_list.sort(key=lambda x: x["score"], reverse=True)
    return new_doc_list
        

def step1_merge_docs(rrf_chunks, web_search_docs):
    merge_list = []
    """
        2个列表的数据格式
        rrf_chunks {
            id:
            score:
            entity: chunk_id,content,file_title,
        }
        web_search_docs:{url,title,hostname,snippet} title:主题,snippet:内容,url:链接
        
        最终需要结果:{
            text: 内容 -》 content,snippet
            title: 主题 -》 title,file_title
            doc_id/chunk_id: 唯一标识,
            url: 链接 -》 url
            - source: 来源标记 ("local" 或 "web")
        }
    """
    for chunk in rrf_chunks:
        temp_dict = {}
        entity = chunk.get("entity") or chunk
        if entity:
            temp_dict["text"] = entity.get("content")
            temp_dict["title"] = entity.get("file_title")
            temp_dict["chunk_id"] = entity.get("chunk_id")
            temp_dict["doc_id"] = None
            temp_dict["source"] = "local"
            temp_dict["url"] = ""
            merge_list.append(temp_dict)
    # 遍历web_search_docs
    for doc in web_search_docs:
        temp_dict = {}
        temp_dict["text"] = doc.get("snippet")
        temp_dict["title"] = doc.get("title")
        temp_dict["doc_id"] = None
        temp_dict["chunk_id"] = None
        temp_dict["source"] = "web"
        temp_dict["url"] = doc.get("url")
        merge_list.append(temp_dict)
    logger.info(f"合并完成: 共 {len(merge_list)} 条")
    return merge_list




if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rerank 本地测试")
    print("="*50)
    
    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "file_title": "算法介绍", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "file_title": "模型介绍", "score": 0.8},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "file_title": "测试文档", "score": 0.1} # 预期低分
    ]
    
    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"} # 预期低分
    ]
    
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？", # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        print("输出结果:", result)
        reranked = result.get("reranked_docs", [])
        
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)
        
        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")
            
        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后
        
        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")

        
     