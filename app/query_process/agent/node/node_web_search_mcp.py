import time
import sys
import json
import asyncio
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from app.core.logger import logger

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_done_task,add_running_task


def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")
    rewritten_query = state.get("rewritten_query", "")
    if rewritten_query:
        query = rewritten_query
    else:
        query = state.get("original_query", "")
    web_search_docs = []
    result = asyncio.run(mcp_call(query))
    if result and result.content:
        print(f"isError:{result.isError}")
        item = result.content[0]
        if item:
            json_str = item.text
            temp = json.loads(json_str)
            pages = temp.get("pages")
            web_search_docs = pages

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    # 调用mcp外部引擎
    print(f"调用外部mcp引擎")

    print("---node-web-search-mcp处理结束---")
    return {"web_search_docs": web_search_docs}


async def mcp_call(query):
    """
    异步调用百炼MCP搜索服务的核心函数。

    该函数负责初始化MCP客户端，建立SSE连接，调用远程工具，并返回原始结果。

    :param query: 搜索查询词（通常是经过改写后的精准Query）
    :return: MCP返回的原始结果对象 (包含 content, isError 等字段)
    """

    # ==================================================================================
    # 初始化百炼MCP SSE客户端
    # ----------------------------------------------------------------------------------
    # MCPServerSse 是一个基于 SSE (Server-Sent Events) 协议的 MCP 客户端实现。
    # 它的作用是连接到阿里云百炼提供的 MCP 服务端点，从而让我们可以像调用本地函数一样调用远程工具。
    #
    # 参数解释：
    # name: 客户端名称，用于日志标识，方便调试。
    # params: 连接配置字典
    #   - url: MCP 服务的 SSE 接口地址 (例如: .../mcps/WebSearch/sse)
    #   - headers: HTTP 请求头，必须包含 Authorization 字段传入 API Key 进行鉴权。
    #   - timeout: 连接建立和整体请求的超时时间。
    #   - sse_read_timeout: 读取 SSE 事件流的超时时间，防止流中断导致挂起。
    # ==================================================================================
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 300,
            "sse_read_timeout": 300
        }
    )

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_config.mcp_base_url}")
        # 建立与MCP服务的SSE连接（异步方法，需await）
        await search_mcp.connect()

        logger.info(f"[MCP] 连接成功，正在调用工具 'bailian_web_search' 查询: {query}")
        # 调用百炼MCP的搜索工具（核心步骤）
        # tool_name: "bailian_web_search" 是百炼官方定义的工具名称
        # arguments: 工具所需的参数，这里需要 "query" (查询词) 和 "count" (返回数量)
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
        )
        logger.info("[MCP] 工具调用完成，已获取返回结果")
        return result

    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)
        return None

    finally:
        # 无论调用成功/失败，最终都关闭MCP连接（释放资源，异步方法）
        await search_mcp.cleanup()

if __name__ == "__main__":
    res = node_web_search_mcp({"original_query": "如何使用百度搜索","session_id":"user_dong","is_stream":False})
    print(res)

