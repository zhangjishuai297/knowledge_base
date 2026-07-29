
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task, set_task_result
from app.utils.sse_utils import create_sse_queue, get_sse_queue, push_to_session, remove_sse_queue
from app.query_process.agent.state import QueryGraphState
import sys

# 最大上下文长度,太多模型会超出限制
MAX_CONTEXT_CHARS = 12000

def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    **1）** **检查答案：** 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出

    **2）** **生成提示词：**根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 

    **3）** **生成答案：**调用大模型输出答案 ，注意判断是否需要流式输出需要则流式输出

    **4）** **保存答案：**把答案写入到mongodb的history中 利用clients/mongo_history_utils.py中的save_chat_message方法

    **5）** **流操作的final push:  做最后一次push操作**（主要是为了触发前端图片渲染) 
    """
    # 返回给前端的输入格式
    # READY = "ready"         # 连接建立
    # PROGRESS = "progress"   # 任务节点进度
    # DELTA = "delta"         # LLM 流式输出增量
    # FINAL = "final"         # 最终完整答案
    # ERROR = "error"         # 错误信息
    # CLOSE = "__close__"     # 关闭连接信号
    logger.info(f"开始处理答案")
    # 记录任务开始
    # 获取session_id
    session_id = state.get("session_id","")
    is_stream = state.get("is_stream",False)
    add_running_task(session_id, sys._getframe().f_code.co_name,is_stream)
    #  1检查答案：** 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    # 在node_item_name_confirm 节点中已经处理了答案,直接回答
    
    answer = state.get("answer","")
    if answer:
        logger.info(f"答案已经存在,直接返回答案:{answer}")
        handle_answer(answer,is_stream,session_id)
        return state
    
    # 如果没有answer,进行下一步处理
    # 2生成提示词：**根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 
    logger.info(f"生成提示词")
    fianl_prompt = generate_prompt(state)
    
    # 生成答案处理处理答案：** ，注意判断是否需要流式输出需要则流式输出
    logger.info(f"使用模型生成答案")
    final_answer = generate_answer(fianl_prompt,is_stream,session_id)
    state['answer'] = final_answer
    
    #  **4）** **保存答案：**把答案写入到mongodb的history中 利用clients/mongo_history_utils.py中的save_chat_message方法


    
    # 记录任务结束
    set_task_result(state["session_id"], "answer", state['answer'])
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    return {"answer": final_answer}

def generate_answer(fianl_prompt,is_stream,session_id):
    llm_client = get_llm_client()
    messages = [
        {"role": "user", "content": fianl_prompt}
    ]
    # 判断是否需要流式输出需要则流式输出
    final_answer = ""
    if is_stream:
        for chunk in llm_client.stream(messages):
            content = getattr(chunk,"content","")
            if content:
                push_to_session(session_id, "delta", {"delta": content})
                final_answer += content
    else:
        # 不是流式输出,直接调用invoke
        res = llm_client.invoke(messages)
        content = res.content
        final_answer += content
    set_task_result(task_id=session_id, key="answer", value=final_answer)
    return final_answer
                

def generate_prompt(state: QueryGraphState):
    """
    # 查看提示词文档中需要哪些参数
    #answer_out.prompt -> 
    # context, 检索出的最终片段
    # history, 历史记录
    # item_names, 主体名称
    # question,用户问题
    """
    reranked_docs= state.get("reranked_docs",[])
    logger.info(f"reranked_docs数量={len(reranked_docs)}")
    chat_history = state.get("history",[])
    logger.info(f"chat_history数量={len(chat_history)}")
    item_names = state.get("item_names",[])
    logger.info(f"item_names数量={len(item_names)}")
    question = state.get("rewritten_query","")
    if not question:
        logger.info("没有重新问题,使用原始问题")
        question = state.get("original_query","")
        
    # 1.处理片段
    context_list = []
    # 计算上下文长度,初始设置为0
    context_len = 0
    
    for index, doc in enumerate(reranked_docs,start=1):
        text = doc.get("text","")
        title = doc.get("title","")
        source = doc.get("source","")
        url = doc.get("url","")
        chunk_id = doc.get("chunk_id")
        score = doc.get("score",0)
        single_context = f"文本{index}详细信息如下\n text:{text} \n"
        single_context += f"title:{title},source:{source},url:{url},chunk_id:{chunk_id},score:{score}"
        if context_len + len(single_context) > MAX_CONTEXT_CHARS:
            break
        context_len += len(single_context)
        context_list.append(single_context)
        
    # 拼接上下文
    context = "\n\n".join(context_list)
    
    # 2.处理历史消息
    history_str = ""
    if chat_history:
        for index, message in enumerate(chat_history,start=1):
            role = message.get("role","")
            text = message.get("text","")
            if role == "user" and text:
                history_str += f"用户: {text}\n"
            elif role == "assistant" and text:
                history_str += f"小助手: {text}\n"
            if context_len + len(history_str) > MAX_CONTEXT_CHARS:
                break
            context_len += len(history_str)
    else:
        history_str = "无历史记录"
        
    # 3.处理item_names
    item_names_str = ",".join(item_names)
    # 处理item_names
    item_names = ",".join(item_names)
    # 处理question,如需处理
    # 把内容添加到模版文件
    final_prompt = load_prompt("answer_out",context=context,history=history_str,item_names=item_names_str,question=question)
    return final_prompt
    
    

def handle_answer(answer,is_stream,session_id):
    """
    param: answer: 输出的答案
    param: is_stream: 是否流式输出
    param: session_id: 会话id
    """
    # 无论是否是流输出
    set_task_result(task_id=session_id, key="answer", value=answer)
    
    # 如果是流输出,和前端约定好的输出格式  push_to_session
    # 如前端监听事件为delta,监听到后,代码中数据通过data.delta获取,这里传入的参数为delta
    # 前端代码如下
    # es.addEventListener('delta', (e) => {
    #       try {
    #         const d = JSON.parse(e.data || '{}');
    #         const delta = d.delta || '';
    #         if (delta) {
    #           rawAnswerText += delta;
    if is_stream:
        push_to_session(session_id, "delta", {"delta": answer})
    
    
    