from pathlib import Path
import inspect
import requests
import os
import time
import shutil
import zipfile
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task,add_done_task
from app.utils.format_utils import format_state
from app.import_process.config.mineru_config import mineru_config
from app.import_process.agent.state import create_default_state



# MinerU配置（缓存配置信息）
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_token
# 全局常量统一管理
MODEL_VERSION = "vlm"
UPLOAD_TIMEOUT = 60 # 上传文件超时时间
DOWNLOAD_TIMEOUT = 120 # 下载zip文件超时时间
POLL_REQ_TIMEOUT = 10
POLL_INTERVAL = 3   # 轮询间隔3秒（平衡查询频率和服务端压力）
MAX_TASK_TIMEOUT = 600 # 最大超时时间10分钟（适配600页内PDF）
GET_URL_TIMEOUT = 30

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
    try:
        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = step_1_validate_paths(state)
        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        # 步骤3：下载ZIP包并提取MD文
        md_path_obj = step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
        # 更新工作流状态
        state["md_path"] = str(md_path_obj.absolute())
        state["is_md_read_enabled"] = True
        state["md_content"] = md_path_obj.read_bytes()
    except Exception as e:
        logger.error(f"【{func_name}】PDF转MD流程执行失败，错误信息：{e}")
        raise
    finally:
        # 结束：记录节点运行状态
        add_done_task(state["task_id"], func_name)
        logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{state}")
        return state
def step_3_download_and_extract(zip_url:str, output_dir:Path, stem:str):
    """
    步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
    1.下载zip
    2.清理旧目录,解压zip
    3.查找MD文件（按优先级：(pdf文件去掉后缀).md > full.md > 其他.md）
    4.重命名统一为PDF同名
    参数:zip_url-待下载的ZIP文件链接,output_dir-输出目录路径对象,stem-待重命名的文件名（去掉后缀）
    返回:最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)

    """
    # 1.下载zip
    
    try:
        logger.info(f"1.[下载ZIP]:开始下载ZIP文件，链接为：{zip_url}")
        response = requests.get(zip_url, timeout=DOWNLOAD_TIMEOUT)
    except Exception as e:
        raise RuntimeError(f"1.[下载ZIP]:失败，错误信息：{e}")
    if response.status_code != 200:
        raise RuntimeError(f"1.[下载ZIP]:失败，错误信息：{response.text}")
    zip_path = output_dir / f"{stem}.zip"
    # 检查是否有旧zip文件
    # missing_ok=True：文件不存在也不报错，静默跳过
    if zip_path.exists():
        zip_path.unlink(missing_ok=True)
        logger.info(f"1.[下载ZIP]:发现旧ZIP文件，已删除,当前状态为：{zip_path.exists()}")
    zip_path.write_bytes(response.content)
    logger.info(f"1.[下载ZIP]:成功，已保存至：{zip_path.absolute()}，大小为：{zip_path.stat().st_size} bytes")

    # 2.清理旧目录,解压zip
    extract_target_dir = output_dir / stem
    
    if extract_target_dir.exists():
        try:
            shutil.rmtree(extract_target_dir)
            logger.info(f"2.[清理旧目录]:成功，已删除旧目录：{extract_target_dir.absolute()}")
            time.sleep(0.5)  # 等待文件系统释放锁，避免mac系统异步bug
            """
            系统：macOS APFS 文件系统，删除是异步后台执行
            Python shutil.rmtree()：调用系统删除接口后立刻返回，不等磁盘真正删完文件
            Python zipfile.extractall() 底层规则：目标文件已存在，直接跳过不解压，无覆盖、无报错
            MinerU 压缩包内部包含大量小文件（你这个包 189 个文件），系统删除回收耗时更长
            """
        except Exception as e:
            logger.error(f"2.[清理旧目录]:失败，错误信息：{e}")
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_target_dir)
    logger.info(f"2.[解压ZIP]:成功，已解压至：{extract_target_dir.absolute()}")

    # 3.查找MD文件（按优先级：(pdf文件去掉后缀).md > full.md > 其他.md）
    md_files = list(extract_target_dir.glob("*.md"))
    logger.info(f"3.[查找MD文件]:找到MD文件共：{len(md_files)}个")
    if not md_files:
        raise FileNotFoundError(f"3.[查找MD文件]:未找到MD文件")
    # 定义接受md文件变量,方便重命名
    target_md_file: Path = None
    for md_file in md_files:
        # 按照原始PDF文件去掉后缀名查找
        if md_file.stem == stem:
            logger.info(f"3.[查找MD文件]:找到目标MD文件：{md_file.absolute()}")
            target_md_file = md_file
            break
        elif md_file.stem == "full":
            logger.info(f"3.[查找MD文件]:找到full.md文件：{md_file.absolute()}")
            target_md_file = md_file
            break
    # 如果未找到目标MD文件，则取列表第一个
    if not target_md_file:
        target_md_file = md_files[0]
        logger.info(f"3.[查找MD文件]:未找到目标MD文件，使用第一个MD文件：{target_md_file.absolute()}")

    # 4.重命名统一为PDF同名
    # 如果md文件名和PDF文件名不一致
    if target_md_file.stem != stem:
        logger.info(f"4.[重命名MD文件]:MD文件名与PDF文件名不一致，开始重命名为：{stem}.md")
        new_stem = target_md_file.with_stem(stem)
        try:
            new_file = target_md_file.rename(new_stem)
            target_md_file = new_file
            logger.info(f"4.[重命名MD文件]:成功，已重命名为：{new_file.absolute()}")
        except Exception as e:
            logger.warning(f"4.[重命名MD文件]:失败，错误信息：{e}")
        md_path_obj = target_md_file.absolute()
        logger.info(f"4.[最终MD路径]:{md_path_obj}")
    return md_path_obj
def step_2_upload_and_poll(pdf_path_obj: Path):
    """
    1.配置校验,url和token是否在.env中配置
    2.获取上传链接,构造请求,获取上传链接
    3.上传PDF文件,调用接口(含重试)
    4.任务轮询,获取解析结果,获取下载zip文件链接
    参数:pdf_path_obj-待上传的PDF文件路径对象,output_dir_obj-输出目录路径对象
    返回值:下载zip文件链接full_zip_url
    """
    func_name = inspect.currentframe().f_code.co_name
    # 配置校验
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError(f"[{func_name}]:MinerU配置不完整,请检查.env文件")
    logger.info(f"[{func_name}]:MinerU配置完整,开始处理文件{pdf_path_obj.name}")
    
    # 创建Session（复用TCP连接，禁用代理避免签名验证失败）
    upload_session = requests.Session()
    upload_session.trust_env = False

    # 获取上传链接
    
    batch_path = '/api/v4/file-urls/batch' #适用于本地文件上传解析的场景，可通过此接口批量申请文件上传链接
    request_header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {MINERU_API_TOKEN}"
}
    get_upload_url = f"{MINERU_BASE_URL}{batch_path}"
    request_data = {
    "files": [
        {"name":pdf_path_obj.name}
    ],
    "model_version":"vlm"
}
    response = upload_session.post(get_upload_url, headers=request_header, json=request_data)
    # 完整的响应数据
    resp_data = response.json()
    # 响应校验:状态码是否为200
    if response.status_code != 200:
        raise RuntimeError(f"[{func_name}]:获取上传链接失败,返回状态码为：{response.status_code},返回内容为：{resp_data}")
    
    # 响应内容code不为0,获取上传链接失败
    if resp_data.get("code") != 0:
        raise RuntimeError(f"[{func_name}]:获取上传链接业务失败,返回内容为：{resp_data}")


    # 3.上传PDF文件,调用接口(含重试)
    # 提取核心数据：上传链接和任务唯一标识
    file_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]
    logger.info(f"[获取上传链接] 成功，batch_id：{batch_id}，上传链接已生成")

    try:
        upload_resp = upload_session.put(file_url, data=pdf_path_obj.read_bytes(),timeout=60)
        if upload_resp.status_code != 200:
            raise RuntimeError(f"[{func_name}]:上传PDF失败,返回状态码为：{upload_resp.status_code},返回内容为：{upload_resp.text}")
        logger.info(f"[上传PDF] 成功，文件名：{pdf_path_obj.name}，大小：{pdf_path_obj.stat().st_size} bytes")
    except Exception as e:
        logger.error(f"[上传PDF] 失败，错误信息：{e}")
    finally:
        upload_session.close()
    # 4.任务轮询,获取解析结果,获取下载zip文件链接
    get_result_url = f"{MINERU_BASE_URL}/api/v4/extract-results/batch/{batch_id}"
    # 轮询
    start_time = time.time()
    logger.info(f"[{func_name}]:[任务轮询]:开始轮询解析结果,轮询间隔为{POLL_INTERVAL}秒,最大超时时间为{MAX_TASK_TIMEOUT}秒,batch_id：{batch_id}")
    while True:
        # 超时检查：超过最大时间直接终止轮询
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_TASK_TIMEOUT:
            raise TimeoutError(f"[{func_name}]:[任务轮询]:超过最大时间限制：{MAX_TASK_TIMEOUT}秒,batch_id：{batch_id}")
        
        # 发起轮询请求，短超时10秒，异常则重试
        try:
            result_resp = upload_session.get(get_result_url,headers=request_header,timeout=POLL_REQ_TIMEOUT)
        except Exception as e:
            logger.warning(f"[{func_name}]:[任务轮询]:请求失败，错误信息：{e},将在{POLL_INTERVAL}秒后重试")
            time.sleep(POLL_INTERVAL)
            continue
        
        # 处理HTTP响应错误：5xx服务端繁忙则重试，其他错误直接抛出
        if result_resp.status_code != 200:
            if 500 <= result_resp.status_code < 600:
                logger.warning(f"[{func_name}]:[任务轮询]:服务器繁忙,返回状态码为：{result_resp.status_code},将在{POLL_INTERVAL}秒后重试")
                time.sleep(POLL_INTERVAL)
                continue
            raise RuntimeError(f"[{func_name}]:[任务轮询]:获取解析结果失败,返回状态码为：{result_resp.status_code},返回内容为：{result_resp.text}")
        
        # 解析轮询结果，校验业务状态
        result_data = result_resp.json()
        if result_data.get("code") != 0:
            raise RuntimeError(f"[{func_name}]:[任务轮询]:获取解析结果业务失败,返回内容为：{result_data}")
        # 获取结果信息
        extract_results = result_data['data']['extract_result']
        
        # 结果为空,继续
        if not extract_results:
            logger.info(f"[{func_name}]:[任务轮询]:当前解析状态为空,已耗时{elapsed_time}秒")
            time.sleep(POLL_INTERVAL)
            continue
        
        # 解析任务状态，分支处理
        result_item = extract_results[0]
        state_status = result_item["state"]
        # 状态1：任务完成，提取ZIP下载链接
        if state_status == "done":
            logger.info(f"[{func_name}]:解析完成,已耗时{elapsed_time}秒,开始提取下载链接")
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError(f"[{func_name}]:[任务轮询]:解析完成但未返回下载链接,返回内容为：{result_item}")
            logger.info(f"[{func_name}]:解析完成,下载链接为：{full_zip_url}")
            return full_zip_url
        # 状态2：任务失败，提取错误信息抛出
        elif state_status == "failed":
            error_message = result_item.get("error_message", "未知错误")
            raise RuntimeError(f"[{func_name}]:[任务轮询]:解析失败,错误信息为：{error_message},返回内容为：{result_item}")
        # 状态3：处理中，实时打印进度（覆盖当前行）
        else:
            logger.info(f"[{func_name}]:[任务轮询]:解析中,当前状态为：{state_status},已耗时{elapsed_time}秒,将在{POLL_INTERVAL}秒后继续轮询")
            time.sleep(POLL_INTERVAL)

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



if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "华为平板 C3 用户指南-(BZD-AL00&AL10&W00,EMUI10.1_01,ZH-CN).pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    state = node_pdf_to_md(test_state)
    # logger.info(f"测试获取状态：{state}")

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")