# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
from http import client
import inspect
import logging
from multiprocessing import context
import os
import re
import stat
import sys
from typing import List, Dict, Any, Tuple

from humanfriendly import length_size_units
from app.conf.milvus_config import milvus_config
# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from fastapi import params
from app.clients.milvus_utils import get_milvus_client
from pymilvus import DataType,MilvusClient
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt
from app.utils.task_utils import add_done_task,add_running_task
from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】商品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别商品/主体名称，并生成双路向量（稠密+稀疏）存入数据库
    后续扩展点：支持多主体识别、增加商品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），必须包含chunks/file_title/task_id
    :return: 更新后的状态字典，新增item_name键，且chunks列表中每个元素新增item_name字段
    """
    func_name = inspect.currentframe().f_code.co_name
    logger.info(f"【{func_name}】节点启动")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)
    try:
        # 1. 获取输入参数,校验参数
        file_title, chucks = step1_get_input_params(state)
        if not chucks:
            return state

        # 2. 构建上下文,截取前N个切片的文本内容拼接成提示词
        context = step2_build_context(chucks)
        # 3. 大模型识别,识别出主题名
        item_name = step3_call_llm(file_title,context)
        # 4. 回填数据
        new_chucks = step4_fill_data(chucks,item_name)
        state["chunks"] = new_chucks
        state["item_name"] = item_name
        # 5. 为主体生成向量,包括稠密向量（BGE-M3）和稀疏向量（归一化处理）
        dense_vector,sparse_vector = step5_generate_embeddings(item_name)
        # 6. 存入Milvus   
        step6_insert_milvus(file_title,item_name,dense_vector,sparse_vector)      
    except Exception as e:
        logger.error(f"【{func_name}】节点发生异常：{e}")
            # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)
    logger.info(f"【{func_name}】节点执行完成")
    return state

def step6_insert_milvus(file_title:str,item_name:str,dense_vector,sparse_vector):
    """
    
    """
    # 1. 获取客户端
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus客户端初始化失败，请检查MILVUS_URL环境变量配置")
        raise Exception("Milvus客户端初始化失败")

    try:
        collection_name = milvus_config.item_name_collection
        # 创建集合,之前判断存在不
        if client.has_collection(collection_name):
            logger.info("集合已存在,无需创建")

        else:
            # 定义字段 
            schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            )
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR,max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            # 创建索引
            index_params = client.prepare_index_params()
            index_params.add_index(
            field_name="dense_vector",
            index_name='dense_vector_index',
            index_type="HNSW",
            metric_type="COSINE",
            params={
            # M: 图中每个节点的最大连接数(常用16-64)
                "M": 16,
            # efConstruction: 构建索引时的搜索范围(越大建索引越慢，但精度越高，常用100-200)
                "efConstruction": 200
                # 不同数据体量的推荐建议(万级)：
                        # 10000 条数据：M=16, efConstruction=200
                        # 50000 条数据：M=32, efConstruction=300
                        # 100000 条数据：M=64, efConstruction=400
            }
            )
            index_params.add_index(
            field_name="sparse_vector",
            index_name='sparse_vector_index',
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            # DAAT_MAXSCORE：稀疏向量检索时，只计算可能得高分的维度，跳过大量0值，速度更快。
            # quantization="none"：稀疏向量里的权重是小数，不做压缩，保证精度不丢。
            params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"}
            )
            
            # 创建集合
            client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
            )
            logger.info("集合创建成功")
            
        # 插入之前删除:根据item_name删除
        if item_name:
            client.load_collection(collection_name)
            del_res = client.delete(
            collection_name=collection_name,
            filter=f"item_name == '{item_name}'"
            )
            logger.info(f"删除数据数量:{del_res.get('delete_count')}")
        # 插入数据,参数:集合名data=[],这里1条也用list封装
        data={
            'file_title': file_title,
            'item_name':item_name
            }
        if dense_vector:
            data['dense_vector']=dense_vector
        if sparse_vector:
            data['sparse_vector']=sparse_vector
        res = client.insert(collection_name, [data])
        # 插入后强制加载集合，确保数据立即可查、Attu可视化界面可见
        client.load_collection(collection_name)
        logger.info(f"数据插入数据:{len(res['ids'])}")
    except Exception as e:
        logger.error(f"向量插入失败:{e}")
    

def step5_generate_embeddings(item_name: str):
    # 判断主体名不为空
    if not item_name:
        logger.warning("主体名为空,无法生成向量")
        # 返回值用2个参数接收,稠密向量和稀疏向量
        return None,None
    try:
        # 这里封装的方法参数为列表,支持批量生成
        vectors = generate_embeddings([item_name])
        if vectors and 'dense' in vectors and 'sparse' in vectors:
            # 返回值是字典,dense中是稠密向量列表,sparse中是稀疏向量列表
            # 稀疏向量归一化处理
            dense_vector=vectors.get("dense")[0]
            sparse_vector=vectors.get("sparse")[0]
            logger.info(f"成功生成向量")
        else:
            logger.warning("向量生成值为空")
            dense_vector,sparse_vector = None,None
    except Exception as e:
        logger.error(f"向量生成失败:{e}")
        dense_vector,sparse_vector = None,None
        
    return dense_vector,sparse_vector
    
def step4_fill_data(chucks: List[Dict[str, Any]],item_name: str):
    logger.info(f"回填数据开始")
    for chuck in chucks:
        chuck["item_name"] = item_name
        
    logger.info(f"回填数据成功,item_name:{item_name}")
    return chucks
def step3_call_llm(file_title: str,context: str) ->str:
    # 获取大模型客户端
    model = get_llm_client()
    logger.info("获取大模型客户端成功")
    # 获取提示词
    prompt = load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt = load_prompt("product_recognition_system")
    # 生成消息
    human_massage = HumanMessage(content=prompt)
    system_message = SystemMessage(content=system_prompt)
    # 调用模型
    logger.info("开始调用大模型")
    resluts = model.invoke([system_message,human_massage])
    item_name = resluts.content
    logger.info(f"大模型返回结果:{item_name}")
    # 没有识别出,用file_title
    if not item_name:
        item_name = file_title
    return item_name
def step2_build_context(chucks: List[Dict[str, Any]]):
    
    context_list = []
    contenx_len = 0
    for chuck in chucks[0:DEFAULT_ITEM_NAME_CHUNK_K]:
        chunk_content = chuck.get("content", "")
        context_list.append(chunk_content)
        contenx_len += len(chunk_content)
        if contenx_len > CONTEXT_TOTAL_MAX_CHARS:
            logger.info("上下文长度超出限制，跳出循环")
            break
    context =  "\n\n".join(context_list)[0:SINGLE_CHUNK_CONTENT_MAX_LEN]
    return context
def step1_get_input_params(state: ImportGraphState):
    file_tile = state.get("file_title", "")
    chucks = state.get("chunks", [])
    if not file_tile:
        logger.warning(f"file_title为空")
    if not chucks:
        logger.warning(f"chunks为空,请检查")
    logger.info(f"获取输入参数成功,file_title:{file_tile},chunks个数:{len(chucks)}")
    return file_tile, chucks





# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        # milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        # if milvus_client and collection_name:
        #     milvus_client.load_collection(collection_name)
            # 检索测试结果
            # item_name = result_state.get('item_name')
            # safe_name = _escape_milvus_string(item_name)
            # res = milvus_client.query(
            #     collection_name=collection_name,
            #     filter=f'item_name=="{safe_name}"',
            #     output_fields=["file_title", "item_name"]
            # # )
            # logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()