import time
from sympy import Q

from app.utils.task_utils import add_done_task, add_running_task, set_task_result
from ..state import QueryGraphState
import sys

def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 导入知识图谱 (node_import_kg)
    为什么叫这个名字: 构建 Knowledge Graph (KG) 并存入 Neo4j。
    未来要实现:
    1. 调用 LLM 从文本中抽取实体 (Entity) 和关系 (Relation)。
    2. 连接 Neo4j 数据库。
    3. 执行 Cypher 语句将图谱数据写入数据库。
    """
    
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 后面会调用大模型，进行逻辑处理
    time.sleep(1)
    # 记录任务结束
    state['answer'] = '这是答案'
    set_task_result(state["session_id"], "answer", state['answer'])
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    return {"answer": "这是答案"}