import inspect
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque
import time

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

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
def _find_image_in_md(md_content: str, image_file: Path, context_length=100) -> Tuple[str, str]:
    """
    用re 匹配图片
    md中图片格式：![图片名称](图片路径)
    <img src="images/xxx.jpg"/>
    """
    # TODO 这里解析的图片最多只会匹配一个,mineru解析结果是命名是uuid.图片,md文档中的也是唯一的
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

def step3_generate_image_summaries(target_images,root_folder,max_requests: int = 9)\
    ->Dict[str,str]:
    """
    param: target_images: step2的返回值(图片名,图片路径,(上,下文))
    param: root_folder:一个文档的根目录
    param: max_requests: 给限流器的最大请求数
    return: dict{}键为图片名，值为图片的摘要
    流程:
    1. 获取视觉大模型
    2. 循环遍历图片,调用模型生成图片摘要
    3.处理模型返回结果,存到dict中
    """
    summaries = {}
    # 获取视觉大模型
    try:  
        vision_llm = get_llm_client(lm_config.vl_model)
        logger.info("视觉大模型初始化成功")
        # 使用限流器工具限制对模型的访问次数
        request_times = deque()
        logger.info(f"开始对图片进行摘要处理")
        logger.info(f"本次需要处理图片数量为：{len(target_images)}")
        
        for idx,(image_name, image_path, image_content) in enumerate(target_images,start=1):
            logger.info(f"已经处理了图片数量为：{idx}")
            apply_api_rate_limit(request_times, max_requests=max_requests)
            # 图片字节base64编码,再解码成大模型能看懂的字符串
            base64_image = base64.b64encode(Path(image_path).read_bytes()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
            # 加载提示词,传入必要参数
            prompt_text = load_prompt \
            ("image_summary",root_folder=root_folder,image_content=image_content)
            # 构造多模态消息
            messages = [
                HumanMessage(
                    content=[
                        # 文本提示词：携带上下文，限定摘要规则
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        # 多模态核心：Base64编码图片数据
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                )
            ]
            # 调用视觉大模型
            response = vision_llm.invoke(messages)
            
            # 返回结果提取,摘要内容
            summary = response.content.strip().replace("\n", "")
            summaries[image_name] = summary
            logger.info(f"{image_name}图片摘要处理成功: {summary}") 
    except Exception as e:
        logger.error(f"{image_path}图片摘要处理失败: {e}")  
    logger.info(f"图片摘要处理完成,共处理{len(summaries)}张图片")
    return summaries

def step4_upload_and_replace(minio_client:Minio,doc_stem:str,summaries:Dict[str,str],md_content:str,targets):
    """
    流程
    1.上传图片到MinIO
    2.合并摘要和url
    3.替换md图片引用
    param:minio_client
    param:doc_stem,文档文件名,上传子目录
    param:targets 待处理图片信息:图片名,图片路径 ,上下文
    param:summaries 图片摘要
    param:md_content 原始md内容
    return:图片引用替换后的新内容
    """
    
    # 构造要上传的目录
    upload_dir = minio_config.minio_img_dir +  "/" + doc_stem
    # 1.上传图片到MinIO
    urls = upload_to_minio(minio_client, upload_dir, targets)
    # 步骤3：合并图片摘要和URL，过滤上传失败的图片
    image_info = {k: (v,urls.get(k)) for k,v in summaries.items() if urls.get(k)}
    logger.info(f'合并成功的图片数量{len(image_info)}')
    # 步骤4：替换MD内容中的本地图片引用为MinIO远程引用
    if image_info:
        md_content = replace_md_img_url(md_content, image_info)

    return md_content
def replace_md_img_url(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    for image_name,(summary,url) in image_info.items():
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        replace = f"![{summary}]({url})"
        md_content = re.sub(pattern=pattern,repl=replace,string=md_content)
    # 输出替换后的内容,只展示前500
    # if len(md_content) > 500:
    #     logger.info(f"[替换图片URL] 替换后的MD内容：{md_content[:500]}")
    # else:
    #     logger.info(f"[替换图片URL] 替换后的MD内容：{md_content}")
    # 返回替换后的MD内容
    return md_content


def upload_to_minio(minio_client: Minio, upload_dir: str, targets:List[Tuple[str, str, Tuple[str, str]]]):
    if not targets:
        return {}
    bucket_name = minio_config.bucket_name
    # 清理目录图片
    # 遍历目录下所有文件,minio中前缀最开始不能有/
    prefix = upload_dir.replace(" ", "")
    try:
        # 遍历前缀下全部对象，构建删除列表
        delete_list = list(map(
            lambda obj: DeleteObject(obj.object_name),
            minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
        ))
        if delete_list:
            logger.info(f"开始清理minio旧文件,待清理的文件数量为:{len(delete_list)},文件前缀为:{prefix}")
        
            # 返回的是DeleteError对象的迭代器
            errors = minio_client.remove_objects(bucket_name, delete_list)
            for err in errors:
                logger.error(f"删除失败: {err}")
        else:
            logger.info(f"minio没有需要清理的文件,文件前缀为:{prefix}")
    except Exception as e:
        logger.error(f"清理minio失败，错误信息：{e}")   
     
    urls = {}   
    # 上传图片
    for image_name,image_path, _ in targets:
        # 上传文件
        try:
            # 解析图片后缀
            suffix = Path(image_path).suffix[1:]
            minio_client.fput_object(
                bucket_name=bucket_name, # 存储桶名称
                object_name=f'{prefix}/{image_name}', # 文件名
                file_path=image_path, # 本地文件路径
                content_type=f"image/{suffix}"  # 图片MIME类型，浏览器正常预览
            )
            image_url = f"{minio_config.endpoint}/{bucket_name}/{prefix}/{image_name}"
            logger.info(f"上传图片开始：{prefix}")
            logger.info(f"上传图片成功：{image_url}")
            # 图片名称 -> 图片URL
            urls[image_name] = image_url
        except Exception as e:
            logger.error(f"上传图片失败：{e}")
        time.sleep(0.02)        
    return urls

def step_5_backup_new_md_file(md_content: str, md_path_obj: Path):
    new_md_path = md_path_obj.with_stem(md_path_obj.stem + "_new")
    new_md_path.write_text(md_content,encoding="utf-8")
    logger.info(f"5.[备份新MD文件]:成功，已备份为：{new_md_path.name}") 
    return str(new_md_path.absolute)

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
    # 3. 调用多模态模型生成图片描述,返回值类型{key=图片文件名, value=图片摘要}
    # 获取图片和md文档的文件夹名
    doc_path = md_path_obj.parent
    doc_stem = doc_path.stem
    summaries = step3_generate_image_summaries(target_images,doc_stem)
    minio_client = get_minio_client()
    new_md_content = step4_upload_and_replace(minio_client,doc_stem,summaries,md_content,target_images)
    new_md_path = step_5_backup_new_md_file(new_md_content,md_path_obj)
    
    # 更新状态
    state["md_content"] = new_md_content
    state["md_path"] = new_md_path
    logger.info(f"更新状态成功,新的md文件路径为：{new_md_path}")


    # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)
    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{state}")
    return state

if __name__ == "__main__":
    state = create_default_state(md_path="/Users/zhangjishuai/code/knowledge_base/output/华为平板 C3 用户指南-(BZD-AL00&AL10&W00,EMUI10.1_01,ZH-CN)/华为平板 C3 用户指南-(BZD-AL00&AL10&W00,EMUI10.1_01,ZH-CN).md")
    node_md_img(state)