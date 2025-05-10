from contextlib import AsyncExitStack
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams


async def create_agent():
    """Gets tools from MCP Server."""
    common_exit_stack = AsyncExitStack()

    local_tools, _ = await MCPToolset.from_server(
        connection_params=SseServerParams(url="http://localhost:8001/sse"),
        async_exit_stack=common_exit_stack,
    )

    agent = Agent(
        name="nlq_agent",
        model="gemini-2.0-flash",
        description="Agent to answer user question by query data from database",
        instruction="""You are Natural Languqge Query Agent that get question from user and fetch the data from database on related table to answer user question.
You need to call get_related_table_description tool first to get related table description, then use table_columns schema to convert user question into valid SQL query to call database_query tool.
Note that the database_query tool can query on specific dataset. If you need to query more than 1 dataset, you need to call database_query multiple times.
The table with the same dataset can be query with database_query tool at the same time""",
        tools=[
            *local_tools,
        ],
    )
    return agent, common_exit_stack


root_agent = create_agent()
