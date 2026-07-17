import inspect
import sys

from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    func_name = inspect.currentframe().f_code.co_name
    logger.info(f"【{func_name}】节点启动")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)
    # 1. 获取客户端
    item_name = state.get("item_name")
    if not item_name:
        logger.error("item_name为空")
        raise Exception("item_name为空")
    chucks = state.get("chunks",[])
    if not chucks and not isinstance(chucks, list):
        raise Exception("chucks数据异常")
    
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus客户端初始化失败，请检查MILVUS_URL环境变量配置")
        raise Exception("Milvus客户端初始化失败")

    try:
        collection_name = milvus_config.chunks_collection
        # 创建集合,之前判断存在不
        if client.has_collection(collection_name):
            logger.info("集合已存在,无需创建")

        else:
            # 定义字段 
            schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            )
            schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True,auto_id=True)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR,max_length=65535)
            schema.add_field(field_name="part", datatype=DataType.INT8)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR,max_length=65535)
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
        file_title = state.get("file_title")
        if item_name:
            client.load_collection(collection_name)
            del_res = client.delete(
            collection_name=collection_name,
            filter="file_title == $ft && item_name == $in",
            filter_params={"ft": file_title, "in": item_name}
            )
            logger.info(f"删除数据数量:{del_res.get('delete_count')}")
        # 插入数据,参数:集合名data=[],这里1条也用list封装
        data_set = []
        for chunk in chucks:
            data_set.append(chunk)
       
        res = client.insert(collection_name, data_set)
        # 插入后强制加载集合，确保数据立即可查、Attu可视化界面可见
        client.load_collection(collection_name)
        # 获取主键id,chuck_id 集合
        ids = res.get("ids")
        # 把id写到状态
        for i, chunk in enumerate(data_set):
            chunk["chunk_id"] = ids[i]
        logger.info(f"数据插入数据:{len(res['ids'])}条")
        
        state["chunks"] = data_set
        
    except Exception as e:
        logger.error(f"向量插入失败:{e}")
    
    
    add_done_task(state["task_id"], func_name)
    logger.info(f"【{func_name}】节点执行完成")
    return state



if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "item_name": "测试项目_Milvus",
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题_1",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            },
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题_2",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":2,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")