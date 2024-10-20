from vanna.openai.openai_chat import OpenAI_Chat
from openai import AzureOpenAI
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore


client = AzureOpenAI(
  azure_endpoint = "https://exquitech-openai-2.openai.azure.com/", 
  api_key="4f00a70876a542a18b30f13570248cdb",  
  api_version="2024-02-01"
)
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        MY_VANNA_MODEL="sqlagent"
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config) # Make sure to put your AzureOpenAI client here

vn = MyVanna(config={'model': 'exq-gpt-35','api_key':'6c96ae51e9f546e397f55b181018b30d'})
vn.connect_to_mssql(odbc_conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER=exqaisqlserver.database.windows.net;DATABASE=SampleDB;UID=aiadmin;PWD=.aisqlpass1') 
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
print(df_information_schema)
plan=vn.get_training_plan_generic(df_information_schema)

vn.train(plan=plan)
import json

with open('database_schema.json', 'r') as file:
    data = json.load(file)
for key,val in data.items():
    vn.train(ddl=f"{key}:{val}")
vn.train()
from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run()

