from dataclasses import dataclass
from typing import List

import sqlite3
import glob
import tqdm
import os
import chromadb
import pickle
from google import genai
import time
from dotenv import load_dotenv

# read value from environment variable or from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
DB_BASE_PATH = os.getenv("DB_BASE_PATH")


# define table data class
@dataclass
class column_info:
    name: str
    type: str


@dataclass
class table_info:
    db_name: str
    db_path: str
    name: str
    columns: List[column_info]


# create table information
table_infos: List[table_info] = []
sqlite_files = glob.glob(os.path.join(DB_BASE_PATH, "database/*/*.sqlite"))
for sqlite_file in tqdm.tqdm(sqlite_files):
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    # list table in dataset
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]

        # get table schemas
        cursor.execute(f"PRAGMA table_info({table_name});")
        col_info = cursor.fetchall()

        db_name = os.path.splitext(os.path.basename(sqlite_file))[0]
        table_infos.append(
            table_info(
                db_name=db_name,
                db_path=sqlite_file,
                name=table_name,
                columns=[column_info(name=col[1], type=col[2]) for col in col_info],
            )
        )

# save table information
with open("table_infos.pkl", "wb") as f:
    pickle.dump(table_infos, f)

# create vector db to file ./chroma
client = chromadb.PersistentClient()
collection = client.create_collection("cosql")

# add embedding to vector db
ai_client = genai.Client(api_key=API_KEY)
for table in tqdm.tqdm(table_infos):
    # create table description
    col_desc = [{"name": col.name, "data_type": col.type} for col in table.columns]
    table_desc = (
        f"""dataset: {table.db_name}, table: {table.name}, table_columns: {col_desc}"""
    )

    # call embedding with 3 retry in case of 429 error resource exhausted
    embed_vector = []
    attempt = 0
    while len(embed_vector) == 0 and attempt < 3:
        try:
            result = ai_client.models.embed_content(
                model="models/text-embedding-004",
                contents=table_desc,
            )
            embed_vector = result.embeddings[0].values
        except Exception as ex:
            time.sleep(60)
            attempt += 1

    # add embedding with metadata to vector db
    collection.add(
        ids=[f"{table.db_name}.{table.name}"],
        embeddings=[embed_vector],
        metadatas=[
            {
                "db_name": table.db_name,
                "db_path": table.db_path,
                "table_name": table.name,
                "table_desc": table_desc,
            }
        ],
    )
