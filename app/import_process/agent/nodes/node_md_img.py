import inspect
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
# from minio import Minio
# from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState,create_default_state
from app.utils.task_utils import add_running_task,add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt


def step1_check_state(state: ImportGraphState) \
    -> Tuple[str, Path, Path]:
    """
    校验内容,并返回校验后的值
    之前通过2个不同的节点进入改节点 node_entry,node_pdf_to_md,如果是从node_entry进入,没有给md_content赋值
    自己传入的.md文档,images文件夹可能不存在或不同名
    """
    md_content = state.get("md_content","")
    md_path = state.get("md_path","")
    md_path_obj= Path(md_path)
    
    if not md_path:
        logger.warning(f"[{inspect.currentframe().f_code.co_name}]: md_path 为空")
        raise ValueError("md_path 为空")
    # 非空校验,md_content为空,重新赋值
    if not md_content:
        logger.info(f"[{inspect.currentframe().f_code.co_name}]: md_content 为空,重新赋值")
        md_content = md_path_obj.read_text()
        # 更新状态,这里更新状态,因为如果images文件夹不存在,会直接retrurn,不会更新状态,保证后面的节点使用
        state["md_content"] = md_content
        
    images_dir_obj = md_path_obj.parent / "images"

    return md_content, md_path_obj,images_dir_obj

def step2_filter_images(images_dir_obj: Path, md_content: str) \
    -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
    param:
        md_content
        images_dir_obj 图片文件夹路径对象
        return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 
        图片上下文)元组,指的是在md文档中,图片的前后引用的文本(上问,下文)
    """
    target_images = []
    img_num = 0
    for image_file in images_dir_obj.iterdir():
        if not image_file.suffix in [".png", ".jpg", ".jpeg", ".gif"]:
            logger.info(f"图片格式不支持: {image_file.name}")
            continue
        # 从md中过滤图片,并返回图片上下文
        img_num += 1
        context_results= _find_image_in_md(md_content, image_file)
        
        # 获取上下文结果的第一条
        if len(context_results) == 0:
            logger.debug(f"未找到该图片的引用,跳过: {image_file.name}")
            continue
        context = context_results[0]
        target_images.append((image_file.name,str(image_file.absolute()),context))
        logger.debug(f"图片加入待处理列表: {image_file.name}")
    
    logger.info(f"图片处理之前的数量: {img_num} 张")
        
    logger.info(f"找到图片: {len(target_images)} 张")
        
    return target_images
        
        
    
# BUG: 这里需要重构图片匹配正则，.md文件中还存在html标签格式的图片
def _find_image_in_md(md_content: str, image_file: Path, context_length=20) -> Tuple[str, str]:
    """
    用re 匹配图片
    md中图片格式：![图片名称](图片路径)
    <img src="images/xxx.jpg"/>
    """

    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file.name) + r"\)")
    pattern2 = re.compile(r"<img src=\".*?" + re.escape(image_file.name) + r"\"/>")
    # 匹配图片名称
    results = [] # 存储上下文的列表
    for m in pattern.finditer(md_content):
        start,end = m.span()
        up_context = md_content[max(0,start-context_length):start]
        down_context = md_content[end:min(len(md_content),end+context_length)]
        results.append((up_context,down_context))
        logger.debug(f"{image_file.name} 匹配成功: 上文:{up_context} 下文:{down_context}")
    
    if len(results) > 1:
        logger.info(f"{image_file.name} 匹配成功: {len(results)} 个结果")
            
    return results
        
    

    

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    流程:
    1. 对md的content,文件路径进行校验,获取image文件夹路径
    2. 遍历image文件夹,在md文件中过滤实际使用到的图片
    3. 调用多模态模型生成图片描述
    4. 把图片上传到MinIO,并替换md文件中的图片链接为MinIO URL,并填充图片描述
    5. 备份原md文件,保存处理后的md文件并更新状态
    params:
        state: ImportGraphState(md_path,md_content等核心信息)
    return:
        state: ImportGraphState(更新后的全局状态新的md_path,md_content等核心信息)
    """
    # 记录当前运行任务，用于任务监控和状态追踪
    # debug级别的日志,只有把.env中的日志级别改成dubug才生效
    func_name = inspect.currentframe().f_code.co_name
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{state}")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)
    
    # 1. 对md的content,文件路径进行校验,获取image文件夹路径
    md_content, md_path_obj,images_dir_obj = step1_check_state(state)
    
    if not images_dir_obj.exists():
        logger.info(f"[{inspect.currentframe().f_code.co_name}]: 图片文件夹不存在,无需处理图片,退出节点:{images_dir_obj.absolute()}")
        return state
    
    # 2. 遍历image文件夹,在md文件中过滤实际使用到的图片
    target_images = step2_filter_images(images_dir_obj, md_content)
    # logger.info(f"获取实际使用的图片列表:{target_images}")

    # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)
    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{state}")
    return state

if __name__ == "__main__":
    state = create_default_state(md_path="/Users/zhangjishuai/code/knowledge_base/output/华为平板 C3 用户指南-(BZD-AL00&AL10&W00,EMUI10.1_01,ZH-CN)/华为平板 C3 用户指南-(BZD-AL00&AL10&W00,EMUI10.1_01,ZH-CN).md")
    node_md_img(state)
