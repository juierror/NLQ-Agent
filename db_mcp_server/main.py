import os
from typing import Any
from fastmcp import FastMCP
import chromadb
import sqlite3
from google import genai
from dotenv import load_dotenv

# read value from environment variable or from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
DB_BASE_PATH = os.getenv("DB_BASE_PATH")

# create server
mcp = FastMCP(name="DB server")

# load vector db
chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_collection(name="cosql")

# create genai client
ai_client = genai.Client(api_key=API_KEY)


@mcp.tool()
def get_related_table_description(query: str) -> list[str]:
    """Get top 4 of related table description

    Args:
        query (str): user question

    Returns:
        list[str]: list of table description in this format ["dataset: <dataset_name>, table: <table_name>, table_columns: [{'name': <column_name>, 'data_type': <column_data_type>},...]]"
    """
    embed_result = ai_client.models.embed_content(
        model="models/text-embedding-004",
        contents=query,
    )
    query_embedding = embed_result.embeddings[0].values
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4,
    )

    return [result["table_desc"] for result in results["metadatas"][0]]


@mcp.tool()
def database_query(dataset: str, sql_query: str) -> list[tuple[Any]]:
    """query dataset to get result

    Args:
        dataset (str): dataset name from get_related_table_description. the sql_query must not contain the dataset name and only contain table_name.
        sql_query (str): sql query on dataset without dataset name in query.

    Returns:
        list[tuple[Any]]: list of each tuple row in this example format [(column_1_value, column_2_value),...]
    """
    db_path = os.path.join(DB_BASE_PATH, f"{dataset}/{dataset}.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()

    return rows


if __name__ == "__main__":
    # run sse server on port 8001
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
