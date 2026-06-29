import sys
import os
import inspect
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState,create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task,add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """
    func_name = inspect.currentframe().f_code.co_name
    # debug级别的日志,只有把.env中的日志级别改成dubug才生效
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")
    
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)
 
    # 1.  **接收状态**: 获取 `local_file_path`。
    local_file_path = state.get('local_file_path','')
    if not local_file_path:
        logger.error(f"[{func_name}]:核心参数确实,未配置local_file_path,文件路径为空")
        return state
    # 2.  **判断类型**: 检查文件后缀是 `.pdf` 还是 `.md`。
    if local_file_path.endswith(".pdf"):
        # 3.  **设置标记**: 更新 state 中的 `is_pdf_read_enabled` 或 `is_md_read_enabled`，供主图路由使用。
        logger.info(f"[{func_name}]:文件类型校验通过,{local_file_path}>PDF类型,开启PDF解析流程")
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        logger.info(f"[{func_name}]:文件类型校验通过,{local_file_path}>MD类型,开启MD解析流程")
        state["is_md_read_enabled"] = True 
        state["md_path"] = local_file_path
    else:
        # 非pdf和md格式
        logger.warning(f"[{func_name}]:文件类型校验失败,{local_file_path}>仅支持MD/PDF格式")


    # 4.  **提取标题**: 从文件名中提取 `file_title`，后续作为元数据。
    # 提取文件末尾的文件或文件夹名
    file_name = os.path.basename(local_file_path)
    file_title = os.path.splitext(file_name)[0]
    state["file_title"] = file_title
    logger.info(f"【{func_name}】文件业务标识提取完成：file_title = {state['file_title']}")
    
    # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)

    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")
    return state


# if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")