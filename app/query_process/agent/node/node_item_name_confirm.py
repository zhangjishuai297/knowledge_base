import json
import re
import time
import sys

from pandas import options
from scipy import sparse
from scipy.fftpack import sc_diff
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.core.load_prompt import load_prompt
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    1. 从mongo中获取历史对话记录
    2. 把历史对话结合用户问题向大模型提问,获取结构数
    {"rewritten_query":"大模型重构的提问,后续代替原是问题","item_names":["商品1","商品2"]}
    3. 把得到的item_names进行向量化,并进行检索匹配 
    4.根据评分整合结 [确认了的item列表] [不确定需要让用户选择item列表] ,评分太低给用户回答,让用户给出item_name
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    
    query = state.get("original_query", "")
    if not query:
        raise Exception("请输入问题")
    # 1. 从mongo中获取历史对话记录
    history_chat = get_recent_messages(state.get("session_id", ""))
     # 2. 把历史对话结合用户问题向大模型提问,获取结构数
    llm_res = step2_use_llm(query, history_chat)
    item_names = llm_res.get("item_names", [])
    state['item_names'] = item_names
    rewritten_query = llm_res.get("rewritten_query", "")
    logger.info(f"[LLM] 模型返回结果：{llm_res}")
    if not item_names:
        state['answer'] = "没有识别到提问主体,请确认"
        return state
    
    
    # 3. 把得到的item_names进行向量化,并进行检索匹配
    extract_list = step3_vector_match(item_names)
    # 4.根据评分整合结果 [确认了的item列表] [不确定需要让用户选择item列表] ,评分太低给用户回答,让用户给出item_name
    comfirm_res = step4_comfirm_item_name(extract_list)
    # 5. 根据整合的结果,决定走向
    new_state = step5_check_item_name_direction(state,comfirm_res)
    # new_state["history"] = json.dump(history_chat,ensure_ascii=False)
    new_state['rewritten_query'] = rewritten_query
    
    
    save_chat_message(session_id=state['session_id'],
                      role = "user", 
                      text= state['original_query'],
                      rewritten_query=rewritten_query, 
                      item_names=new_state.get('item_names',[]))
    
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
   

    print(f"---node_item_name_confirm---处理结束")

    return new_state

def step5_check_item_name_direction(state: ImportGraphState,comfirm_res):
    confirmed_item_names = comfirm_res.get('confirmed_item_names', [])
    options = comfirm_res.get('options', [])
    if len(confirmed_item_names) > 0:
       state['item_names']  = confirmed_item_names
       return state
    if len(options) > 0:
        state['item_names'] = []
        state['answer'] = f"请确认你想咨询的商品的名称,请选择:{options}"
        return state
    
    state['item_names'] = []
    state['answer'] = f"未找到商品名称,请重新输入"
    return state
def step4_comfirm_item_name(extract_list):
    # 根据评分整合结果,
    # >0.85 进入确认列表
    # 0.65~0.85 进入选择列表
    # <0.65 不处理
    comfirm_list = []
    optional_list = []
    for item in extract_list:
        # 获取一组的匹配值
        matches = item.get("matches",[])
        extracted_name = item.get("extracted_name","")
        if not matches:
            logger.info(f"{extracted_name}无匹配结果,跳过")
            continue
        logger.info(f"正在对{extracted_name}检索到的匹配结果进行整合")
        # 临时确认list
        temp_confirm_list = []
        # 临时可选list
        temp_optional_list = []
        for match in matches:
            score = match.get("score",0.0)
            item_name = match.get("item_name","")
            # 如果评分大于0.85，则加入确认列表,无需在判断后面的评分
            if score > 0.85:
                temp_confirm_list.append(item_name)
            elif score > 0.65:
                temp_optional_list.append(item_name)
        # 判断有没有高分
        has_high = len(temp_confirm_list) > 0
      
        if has_high:
            if len(temp_confirm_list)==1:
                comfirm_list.append(temp_confirm_list[0])
            if len(temp_confirm_list) > 1:
                # 判断匹配到的item_name和extracted_name是否相同,如果相同,则加入确认列表,直接添加到确认列表
                # 如果不同,则直接取临时list的第一个
                # 是否命中extracted_name
                is_hit = False
                for temp_item in temp_confirm_list:
                    if temp_item == extracted_name and extracted_name:
                        comfirm_list.append(temp_item)
                        is_hit = True
                        break
                if not is_hit:
                    comfirm_list.append(temp_confirm_list[0])
            continue
        if len(temp_optional_list) > 0:
            optional_list.extend(temp_optional_list[:5])  
    
    return {"confirmed_item_names":list(set(comfirm_list)),
            "options":list(set(optional_list))}    
def step3_vector_match(item_names):
    # 获取Milvus客户端
    milvus_client = get_milvus_client()
    #调用嵌入模型,进行向量化
    res = generate_embeddings(item_names)
    dense_vectors = res.get("dense", [])
    sparse_vectors = res.get("sparse", [])
    extract_list = [] # 存放匹配结果{extracted_name:大模型查到的item名称,matches:{"item_name":匹配到的item名称,score:匹配的分数"}}
    for index, item_name in enumerate(item_names):
        # 获取对应item_name对应的向量
        dense_vector = dense_vectors[index]
        sparse_vector = sparse_vectors[index]
        # 进行向量检索,混合检索
        # 创建混合搜索请求对象
        reqs = create_hybrid_search_requests(dense_vector=dense_vector,sparse_vector=sparse_vector)
        # 进行混合搜索
        hybrid_result = hybrid_search(client=milvus_client,
                      collection_name=milvus_config.item_name_collection,
                      reqs=reqs,
                      ranker_weights=(0.8,0.2),
                      norm_score=True)
        
        # 这里加入参数output_filed会有bug,版本问题,先查主键,通过主键查询
        # 2. 提取所有主键
        pk_ids = []
        for hits in hybrid_result:
            for hit in hits:
                pk_ids.append(hit["id"])

        # 3. 通过query一次性取出需要的标量字段
        if pk_ids:
            expr = f"pk in {pk_ids}"
            scalar_data = milvus_client.query(
                collection_name=milvus_config.item_name_collection,
                filter=expr,
                output_fields=["item_name"]
            )
    # 自己写逻辑把 向量检索分数 和 scalar_data 根据pk合并
        
        matches = [] # 存放一组的检索结果
        if hybrid_result:
            for hit in hybrid_result[0]:
                score = hit.get("distance")
                entity = hit.get("entity")
                matches.append({"item_name":entity.get("item_name", ""),"score":score})
        extract_list.append({"extracted_name":item_name,"matches":matches})
        
    return extract_list
        
    
def step2_use_llm(query, history_chat):
      # 遍历列表,把历史对话中的信息,拼接成字符串,对应提示词文件中的{history_text}
    # 存入mongozhong的历史对话格式
        # session_id: str
        # role: str,
        # text: str,
        # rewritten_query: str = "",
        # item_names: List[str] = None,
        # message_id: str = None
    history_text = ""
    for chat in history_chat:
        role = chat.get("role", "")
        text = chat.get("text", "")
        rewritten_query = chat.get("rewritten_query", "")
        item_names = chat.get("item_names", [])
        history_text += f"聊天角色：{role}，回答内容： {text}，重写问题： {rewritten_query}，关联主体： {','.join(item_names)},时间： {chat['ts']}\n"

    logger.info(f"[LLM] 模型输入参数：{history_text}")
    
    # 加载提示词文件
    prompt = load_prompt("rewritten_query_and_itemnames",query=query,history_text=history_text)
    messages = [
        {"role": "user", "content": prompt},
    ]
    # 大模型实例
    llm_client = get_llm_client(json_mode=True)
    model_res = llm_client.invoke(messages)
    content = model_res.content
    # 如果返回的是markdown代码块,去掉格式
    # 先判断是否存在代码块标记（辅助）
    if "```json" in content and "```" in content:
    # 正则统一剥离
        content = re.sub(r"```\s*json\s*\n(.*?)\n\s*```", r"\1", content, flags=re.DOTALL).strip()
        logger.info(f"正则匹配后的结果:{content}")
    # 把大模型返回的json数据,解析成字典
    json_res = json.loads(content)
    return json_res


if __name__ == "__main__":
    node_item_name_confirm({"original_query": "华为擎云W585多少钱","session_id":"user_dong","is_stream":False})