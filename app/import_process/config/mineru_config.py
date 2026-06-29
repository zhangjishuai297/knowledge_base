# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()

# 定义minerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_token : str

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_token=os.getenv("MINERU_API_TOKEN")
)


if __name__ == "__main__": 
    try:
        from mineru.client import MinerU

        # 初始化客户端
        client = MinerU(token=mineru_config.api_token)
        
        # 转换本地PDF文件，直接输出md文本
        result = client.extract(
            "/Users/zhangjishuai/code/knowledge_base/doc/华为平板 C3 用户指南-(BZD-AL00&AL10&W00,EMUI10.1_01,ZH-CN).pdf",    # 本地pdf路径
            model="vlm",               # vlm高精度；pipeline快速
            ocr=True,                  # 扫描件自动OCR
            formula=True,              # 识别数学公式为LaTeX
            table=True,                # 还原复杂表格
            language="ch",             # 中文文档
            timeout=600                # 超时10分钟
        )

        # 写入md文件
        md_content = result.markdown
        with open("output.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        # 附带结构化数据（RAG可用）
        # 图片资源链接列表
        # image_list = result.images
        #  # -------------------------- 图片保存逻辑写在这里 --------------------------
        # # 创建images目录，不存在则自动生成
        # img_dir = "images"
        # os.makedirs(img_dir, exist_ok=True)

        # for img_obj in image_list:
        #     # img_obj.path 就是相对路径 images/xxx.jpg
        #     save_full_path = os.path.abspath(img_obj.path)
        #     # 写入二进制图片数据
        #     with open(save_full_path, "wb") as img_file:
        #         img_file.write(img_obj.data)
        # print(f"全部图片已保存至项目根目录 {os.path.abspath(img_dir)}")
    except ImportError:
        print("mineru_open_sdk module not found. Please install it using pip.")
        print("Example: pip install mineru-open-sdk")