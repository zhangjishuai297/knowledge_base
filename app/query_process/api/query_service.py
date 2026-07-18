from pathlib import Path
from turtle import st, update
import uuid
from huggingface_hub import run_as_future
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 返回chat.html页面
@app.get("/")
def chat_html():
    chat_path = Path(PROJECT_ROOT) / "app/query_process/page/chat.html"
    if not chat_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(chat_path,media_type="text/html")

# 定义查询数据结构,做数据校验
class QueryData(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")
    
@app.post("/query")
async def query(background_tasks: BackgroundTasks,request:QueryData):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    """
    query = request.query
    session_id = request.session_id
    is_stream = request.is_stream
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not session_id:
        session_id = str(uuid.uuid4())
    if is_stream:
        # 流式返回,先穿件一个队列
        stream_queue = create_sse_queue(session_id)

        logger.info(f"队列创建成功,stream_queue:{stream_queue}")
        background_tasks.add_task(run_query_graph, query, session_id, is_stream)
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }
    else:
        run_query_graph(query,session_id,is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "done_list":[]
        }

# 后台garph执行逻辑,在query接口中解析的参数传递到这里
def run_query_graph(query: str, session_id: str, is_stream: bool):
    state :QueryGraphState= {
        "session_id": session_id,
        "is_stream": is_stream,
        "original_query": query
    }
    logger.info(f"开始执行查询任务...,获取到的任务参数:{state}")
    try:
        # 更新任务状态
        update_task_status(session_id, TASK_STATUS_PROCESSING,push_queue=is_stream)
        query_app.invoke(state)
        # 更新任务状态
        update_task_status(session_id,TASK_STATUS_COMPLETED,push_queue=is_stream)
    except Exception as e:
        # 任务异常
        update_task_status(session_id,TASK_STATUS_FAILED,push_queue=is_stream)
        # 由于异常,update_task_status中推送的消息状态是固定的progress,但是前端需要的是ERROR
        if is_stream:
            push_to_session(session_id,SSEEvent.ERROR,{"error":str(e)})
        
    
# 健康状态检测  
@app.get("/health")
def health():
    path = Path(PROJECT_ROOT) / "app/query_process/page/query_monitor.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "ok"}

# 流式返回结果
@app.get("/stream/{session_id}")
def stream_result(session_id: str,request: Request):
    return StreamingResponse(sse_generator(session_id,request), media_type="text/event-stream")
    
    
# 查询历史记录接口
@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """
    1 获取历史记录
    2 返回结果
    """
    items = get_recent_messages(session_id)
    return {"session_id": session_id, "items": items}

# 清空历史记录接口
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    1 清空历史记录
    2 返回结果
    """
    result = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)