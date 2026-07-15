"""
项目目标,通过文件上传，将文件导入知识库
1.output目录生成格式 /output/日期(例:2026-06-01)/uuid/
"""
# 1.引入依赖和环境配置
import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
from numpy import add
import uvicorn

# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import create_default_state, get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具
# 2.应用初始化与跨域配置
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)
# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)
# 访问前端页面接口,服务启动后就可以访问,不用单独找页面
@app.get("/import.html")
async def import_page():
    import_html = PROJECT_ROOT / 'app' / 'import_process' / 'page' / 'import.html'
    logger.info(f"import_html类型: {type(import_html)}")
    logger.info(f"import_html路径: {import_html}")
    if not import_html.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=import_html,media_type="text/html")
    
# 3. 后台任务逻辑,langgraph导入逻辑
def graph_task(task_id: str,local_dir:str, local_file_path:str):
    """
    后台任务逻辑,langgraph导入逻辑
    :param task_id:
    :return:
    """
    try:
        # 初始化任务状态
        update_task_status(task_id, "processing")
        logger.info(f"任务开始: {task_id}")
        state = create_default_state(task_id=task_id,local_file_path=local_file_path,local_dir=local_dir)
        # 流式输出返回键值对:节点名,状态值
        for chuck in kb_import_app.stream(state,stream_mode='updates'):
            for node_name, update in chuck.items():
                logger.info(f"节点: {node_name},执行完成")
                # logger.info(f"节点: 更新值 {update}")
        # 更新任务状态
        update_task_status(task_id, "completed")
        logger.info(f"任务完成: {task_id}")
    except Exception as e:
        # 更新任务状态为失败
        update_task_status(task_id, "failed")
        logger.error(f"任务失败: {task_id}")
        logger.error(e)
# 4.文件上传接口
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应"""
    today = datetime.now().strftime("%Y-%m-%d")
    task_ids = []
    # 遍历文件列表,一个文件一个task_id
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"开始处理文件: {file.filename},文件类型: {file.content_type}")
        # 加入任务列表
        add_running_task(task_id, "upload_file")
        local_dir = PROJECT_ROOT / 'output' / today / task_id
        # 判断目录是否存在,不存在则创建
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
        # 创建文件对象
        local_file_path = local_dir / file.filename
        # 把文件内容写入文件对象
        local_file_path.write_bytes(await file.read())
        
        
        add_done_task(task_id, "upload_file")
        logger.info(f"文件保存成功: {local_file_path}")
    
    
    # 开启任务,fastapi后台异步任务,方法,参数...
        logger.info(f"启动后台任务: {task_id}")
        background_tasks.add_task(graph_task,task_id,str(local_dir.absolute()),str(local_file_path.absolute()))
    
    return {
        "message": "上传成功",
        "task_ids": task_ids,
        "code":200
    }

# 4.任务状态查询接口
# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )