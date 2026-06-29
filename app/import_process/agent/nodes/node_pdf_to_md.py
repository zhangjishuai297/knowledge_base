from pathlib import Path
import inspect
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task,add_done_task
from app.utils.format_utils import format_state

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
     LangGraph工作流节点：PDF转MD核心处理节点
    核心流程：路径校验 → MinerU上传解析 → 结果下载解压 → 读取MD内容并更新工作流状态
    参数：state-工作流状态对象，需包含pdf_path/local_dir/task_id
    返回：更新后的工作流状态，新增md_path/md_content
    """
    func_name = inspect.currentframe().f_code.co_name
    # debug级别的日志,只有把.env中的日志级别改成dubug才生效
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)
    # 步骤1：校验PDF路径和输出目录
    pdf_path_obj, output_dir_obj = step_1_validate_paths(state)
    # 步骤2：上传PDF至MinerU并轮询解析结果
    
    
    # 步骤3：下载ZIP包并提取MD文
    
    
    
     # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)
    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")
    return state


def step_1_validate_paths(state):
    func_name = inspect.currentframe().f_code.co_name
    pdf_path = state.get('pdf_path','')
    local_dir = state.get('local_dir','')
    # 非空校验
    if not pdf_path:
        raise ValueError(f"[{func_name}]:核心参数确实,未配置pdf_path,当前值为：{repr(pdf_path)}")
    if not local_dir:
        raise ValueError(f"[{func_name}]:核心参数确实,未配置local_dir,当前值为：{repr(local_dir)}")
    
    # 转换为Path对象统一处理路径
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)
    
    # pdf_path_obj校验文件存在性,校验文件类型非文件加
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"[{func_name}]:pdf_path_obj文件不存在,当前值为：{pdf_path_obj.absolute()}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"[{func_name}]:pdf_path_obj文件类型非文件,当前值为：{pdf_path_obj.absolute()}")
    
    # output_dir_obj 校验目录存在性,不存在则创建
    if not output_dir_obj.exists():
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{func_name}]:创建目录成功,目录为：{output_dir_obj.absolute()}")
    return pdf_path_obj,output_dir_obj