# NLQ Agent

NLQ Agent with ADK and MCP. This agent use [CoSQL Dataset](https://yale-lily.github.io/cosql). Please download it first.

## How to run
```bash
# install uv if you don't have it
pip install uv

# install requirements package
uv sync

# create vector database
cd db_mcp_server
export API_KEY="ai studio api key"
export DB_BASE_PATH="path to root of cosql dataset"
uv run create_vector_db.py

# start MCP Server
cd db_mcp_server
export API_KEY="ai studio api key"
export DB_BASE_PATH="path to root of cosql dataset"
uv run main.py

# open another terminal and run adk
adk web
```