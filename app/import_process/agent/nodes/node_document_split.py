import json
import inspect
import re
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task,add_done_task


# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 300 # 512 - 1500 token
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 400 # 最小的长度


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # debug级别的日志,只有把.env中的日志级别改成dubug才生效
    func_name = inspect.currentframe().f_code.co_name
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{state}")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)
    
    #1. 数据获取,数据校验 获取md_content,file_title
    md_content,file_title = step1_check_value(state)
    
    #2. 标题初切,markdown文档标题切分
    sections,title_count,line_num = step2_split_title(md_content,file_title)
    #3. 兜底,md文档无一个标题
    sections = step3_no_title_header(sections,title_count,md_content,file_title)
    #4. 对过长的chunk进行切分(同一标题,parent_title相同), 之后对过于细碎的合并
    sections = step4_refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
    #5. 打印统计,
    #6. 更新state,备份结果到本地:输出json格式文件[{},{},{}],切片是列表的形式存储,{}里是一个切片的具体信息
    step6_backup_result(sections,state)
    state['chunks'] = sections
    
    
    
    
    
    
    # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)
    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{state}")
    return state
def step6_backup_result(sections: list, state: ImportGraphState):
    md_path = state["md_path"]
    json_file_path = Path(md_path).with_suffix(".json")
    Path(json_file_path).write_text(json.dumps(sections, ensure_ascii=False, indent=4))
def step4_refine_chunks(sections: list, max_length: int = DEFAULT_MAX_CONTENT_LENGTH, min_length: int = MIN_CONTENT_LENGTH):
     # 判断最大长度是否小于等于0,判断是否有有效的值
    final_sections = []
    if not max_length or max_length <= 0:
        logger.warning(f"max_length为空或者小于等于0,配置无效,不进行处理！")
        return sections
    
    for section in sections:
        # 切分长文本
        sub_sections = split_long_section(section, max_length)
        final_sections.extend(sub_sections)
    # 合并短chunk
    final_sections = merge_short_chunks(final_sections, min_length)
    # 补全缺失字段,有的元素,没有part 和parrent_title字段
    for section in final_sections:
        section["part"] = section.get("part") or "1"
        section["parent_title"] = section.get("parent_title") or section.get("title") or "无标题"
    return final_sections
    
    
        
def merge_short_chunks(final_sections:list, min_length: int):
    # 定义一个变量,记录当前循环到哪个章节,每当合并到大于min_length时,重新赋值pre_section,确保从新的章节开始合并
    sections = []
    pre_section = None
    for section in  final_sections:
        # 按照指针的方式遍历,将短chunk合并
        # 判断长度是否小于等于min_length,
        # 如果小于,取下一个chuck长度累计,如果大于,不处理
        if not pre_section:
            # 如果是空的,先记录,在进行下一次循环,然后判断
            pre_section = section
            continue
        
        pre_content = pre_section.get("content")
        content = section.get("content")
        
         # 判父标题是否一致且都不为空,一致时判断内容长度,不一致时,把旧的添加到列表,更新pre_section为当前的
        is_same_parent_title = pre_section.get("parent_title") == section.get("parent_title") and section.get("parent_title")
        if len(pre_content) >= min_length or not is_same_parent_title:
            # 如果长度大于min_length,则直接添加到列表,更新pre_section为None
            sections.append(pre_section)
            pre_section = section
            continue
       
       
        if len(pre_content) + len(content) >= min_length:
            # 如果长度小于等于max_length,则合并,更新pre_section
            pre_section["content"] += "\n" + content
            sections.append(pre_section)
            pre_section = None
            continue
        else:
            # 如果文本累加的长度小于min_length,则合并,更新pre_section里面的值,但不追加到列表
            pre_section["content"] += content
            pre_section["part"] = section.get("part")
            
    if pre_section:
        sections.append(pre_section)
        
    return sections
def split_long_section(section: dict, max_length:int):
    """
    功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
    切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
    :param sections: 原始章节字典，必须包含content键，可选title/file_title等
    :param max_length: 单个Chunk最大字符长度，默认使用全局配置
    :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息,parent_title,part,添加的新key
    """
   
    # 定义切割规则
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,        # 单块最大字符
        chunk_overlap=50,      # 块重叠，防止上下文断裂
        separators=["\n\n", "\n", "。", "，", " ", ""] # 中文适配分隔符
    )
    # 定义新的章节列表
    new_sections = []

    # 取出文本,这里的文本不会为空
    content = section["content"]
    if len(content) <= max_length:
        # 长度小于等于max_length,直接返回原数据
      return [section]     
        
    # 切分后的文本列表
    chunks = splitter.split_text(content)
    for i, chunk in enumerate(chunks,start=1):
        if not (text:= chunk.strip()):
            continue
        # 创建新的章节字典
        new_section = {
            "title": f"{section['title']}_{i}",
            "content": text,
            "file_title": section["file_title"],
            "parent_title": section["title"],
            "part": i
        }
            # 添加到新列表中
        new_sections.append(new_section)
    return new_sections
    

def step3_no_title_header(sections: list,title_count: int,md_content: str,file_title: str):
    if title_count == 0:
        logger.info(f"没有找到标题，将使用文件名作为标题")
        # 返回新的章节列表
            # 没有标题
    sections[0]["title"] = sections[0].get("title") or '无标题'
    logger.info(f"找到标题我,文档的标题个数为：{title_count}")
    # 重新返回列表
    return sections

def step2_split_title(content: str, file_title: str):
    """
    param: content 标准化后的md完整内容
    param: title 文件标题,标记章节归属
    return:章节列表[{章节信息:标题,内容},{}], 有效标题数量, 原始文本总行数
    """
    lines = content.split("\n")
    # 识别md文档中是标题的正则表达式
    # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    # #{1,6}：匹配1-6个#（对应MD1-6级标题）
    # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    # .+：标题文字至少1个字符（避免空标题）
    pattern = re.compile(r'^\s*#{1,6}\s+.+')
    # 记录一组标题和内容的临时变量,在读到新标题时,将上一组标题和内容保存到章节列表中,并重置临时变量
    sections = [] # 章节列表
    temp_title = ""
    # 读到的行内容先存储到临时列表变量中,不直接用字符串拼接
    temp_content = []
    # 记录有效标题数量
    title_count = 0
    # 是否是代码块
    is_code_block = False
    
    # 把临时的内容保存到章节列表中
    def _flush_section():
        # 判断标题是否为空,如果为空,说明是第一个标题,不需要刷新内容
        # 这里改为不判断temp_title,因为第一个标题的标题可能为空,有的文档第一个标题在内容下面
        if not temp_content:
            return
        section = {
            "title": temp_title,
            "content": "\n".join(temp_content),
            "file_title": file_title
        }
        sections.append(section)

    # 按行读取
    for i, line in enumerate(lines):
        strip_line = line.strip()
        # 判断是否在代码块内,只更改标志位和追加内容
        # 如开始是False,读到一次,则标记为代码块,再读到一次,则取消代码块标记,只要读到```,改行不是标题,内容追加到temp_content

        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            is_code_block = not is_code_block
            temp_content.append(line)
            continue
        
        is_title = pattern.match(line)
        # 判断是否是标题行,且判断是否在代码块
        if is_title and not is_code_block:
            # 把上一个标题的章节内容刷新到sections中,并重置临时变量
            _flush_section()
            temp_title = strip_line
            temp_content = [temp_title]
            title_count += 1
        else:
            # 不是标题行,把行内容追加到临时变量中
            temp_content.append(line)
            
    # 兜底,把最后一个标题的章节内容刷新到sections中
    _flush_section()
    # 返回内容
    return sections, title_count,len(lines)
        
            
    
        
def step1_check_value(state: ImportGraphState):
    md_content,file_title = state['md_content'], state.get('file_title','Unknown')

    # 非空判断
    if not md_content:
        logger.warning(f"md_content为空")
        md_path = state.get('md_path')
        md_content = Path(md_path).read_text(encoding='utf-8')
        # raise ValueError("md_content为空")
        
    # 统一换行
    md_content.replace("\r\n","\n").replace("\r","\n")
    return md_content,file_title


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/万用表RS-12的使用", "万用表RS-12的使用.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "万用表RS-12的使用",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        # result_state = node_md_img(test_state)
        # logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk")
