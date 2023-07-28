from dotenv import load_dotenv, find_dotenv
import os
import boto3
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Manual input
input_file_1 = 'data/state_of_the_union.txt'
input_file_2 = 'data/Donald J. Trump [February 05, 2019].txt'
input_file = input_file_1  # Input file switching

index = 'aoss_qa'
embedding_dimension = 1536

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Amazon OpenSearch Service connection
host = os.environ['AOSS_ENDPOINT']
splitted_host_str = host.split('.')
region = splitted_host_str[-4]
service = splitted_host_str[-3]
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)
aoss_client_conf = {
    "http_auth": auth,
    "use_ssl": True,
    "verify_certs": True,
    "connection_class": RequestsHttpConnection,
    "pool_maxsize": 20,
    "is_aoss": service == 'aoss'
}

# Index settings and mappings
index_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "knn": True
    },
    "mappings": {
        "properties": {
            "vector_field": {
                "type": "knn_vector",
                "dimension": embedding_dimension,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "l2",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            }
        }
    }
}

# LangChain
text_splitter_conf = {
    "chunk_size": 1000,
    "chunk_overlap": 20,
    "length_function": len,
    "add_start_index": True,
}

# Load document and split to texts
loader = UnstructuredFileLoader(input_file)
text_splitter = RecursiveCharacterTextSplitter(**text_splitter_conf)
texts = loader.load_and_split(text_splitter)

# Connect to OpenSearch
embeddings_model = OpenAIEmbeddings()
vectorstore = OpenSearchVectorSearch(
    opensearch_url=host,
    index_name=index,
    embedding_function=embeddings_model,
    **aoss_client_conf
)

# Create index if not exists
if not vectorstore.client.indices.exists(index):
    vectorstore.client.indices.create(
        index=index,
        body=index_body
    )

# Create unique ids for each text using document metadata
ids = []
for text in texts:
    ids.append('_'.join([str(v) for v in text.metadata.values()]))

# Load text & embeddings to OpenSearch
text_id_list = vectorstore.add_documents(
    texts,
    ids=ids
)
assert len(text_id_list) == len(texts)

# QA
question = "What did the president say about Ketanji Brown Jackson"
# result = vectorstore.similarity_search(question, k=4)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever()
)
print(qa_chain({"query": question}))
