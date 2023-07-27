import os
import boto3
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Parameters - AOSS
host = os.environ['AOSS_ENDPOINT']
input_file = 'data/state_of_the_union.txt'
# input_file = 'data/Donald J. Trump [February 05, 2019].txt'
region = 'eu-west-1'
service = 'es'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)
aoss_client_conf = {
    "http_auth": auth,
    "use_ssl": True,
    "verify_certs": True,
    "connection_class": RequestsHttpConnection,
    "pool_maxsize": 20
}
index = 'aoss_qa'

# Parameters - LangChain
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

# Connect to AOSS
embeddings_model = OpenAIEmbeddings()
aoss = OpenSearchVectorSearch(
    opensearch_url=host,
    index_name=index,
    embedding_function=embeddings_model,
    **aoss_client_conf
)
assert aoss.client.ping()

# Load text & embeddings to AOSS
# Thoughts:
# - Add documents to an existing index without mapping will result in wrong type
# - Add documents and create new index will use default no. of shards & replicas
# - New IDs are generated for each run, introducing duplicated docs in AOSS
#   Try https://github.com/langchain-ai/langchain/blob/4da43f77e5bf3d25f5b7ece8bcba1ab7c6a9abb2/libs/langchain/langchain/vectorstores/opensearch_vector_search.py#L402C13-L402C16
text_id_list = aoss.add_documents(
    texts,
    engine="faiss",
    space_type="l2",
    ef_construction=512,
    m=16
)
assert len(text_id_list) == len(texts)

# # Delete index
# aoss.client.indices.delete(index)

# # Check index status
# print(aoss.client.indices.stats(index))

# # Check index mapping
# print(aoss.client.indices.get_mapping(index))

# # Check first item
# print(aoss.client.search(index=index, size=1))

# QA
question = "What did the president say about Ketanji Brown Jackson"
# result = aoss.similarity_search(question, k=4)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=aoss.as_retriever()
)
qa_chain({"query": question})
