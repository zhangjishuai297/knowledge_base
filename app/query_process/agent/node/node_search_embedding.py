import time
from app.utils.task_utils import add_running_task,add_done_task
from ..state import QueryGraphState
import sys

def node_search_embedding(state: QueryGraphState):
  
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 后面会调用大模型，进行逻辑处理
    time.sleep(1)
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    return {"embedding_chunks":[]}