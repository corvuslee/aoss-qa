import os
import boto3
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# ES/AOSS connection
es_host = os.environ['ES_ENDPOINT']
aoss_host = os.environ['AOSS_ENDPOINT']
credentials = boto3.Session().get_credentials()
es_auth = AWSV4SignerAuth(credentials, 'eu-west-1', 'es')
aoss_auth = AWSV4SignerAuth(credentials, 'eu-west-1', 'aoss')
es_client_conf = {
    "http_auth": es_auth,
    "use_ssl": True,
    "verify_certs": True,
    "connection_class": RequestsHttpConnection,
    "pool_maxsize": 20
}
aoss_client_conf = {
    "http_auth": aoss_auth,
    "use_ssl": True,
    "verify_certs": True,
    "connection_class": RequestsHttpConnection,
    "pool_maxsize": 20
}

# Documents, index
index = 'aoss_qa'
input_file_1 = 'data/state_of_the_union.txt'
input_file_2 = 'data/Donald J. Trump [February 05, 2019].txt'

# LangChain
text_splitter_conf = {
    "chunk_size": 1000,
    "chunk_overlap": 20,
    "length_function": len,
    "add_start_index": True,
}

# Switches
host = es_host  # or aoss_host
input_file = input_file_1

# Load document and split to texts
loader = UnstructuredFileLoader(input_file)
text_splitter = RecursiveCharacterTextSplitter(**text_splitter_conf)
texts = loader.load_and_split(text_splitter)

# Connect to OpenSearch
is_aoss = True if host == aoss_host else False
client_conf = aoss_client_conf if is_aoss else es_client_conf

embeddings_model = OpenAIEmbeddings()
vectorstore = OpenSearchVectorSearch(
    opensearch_url=host,
    index_name=index,
    embedding_function=embeddings_model,
    is_aoss=is_aoss,
    **client_conf
)

# Create unique ids for each text using filename and start index
ids = [
    f"{os.path.basename(text.metadata['source'])}_{text.metadata['start_index']}"
    for text in texts
]
# Load text & embeddings to OpenSearch
text_id_list = vectorstore.add_documents(
    texts,
    ids=ids,
    engine="faiss",
    space_type="l2",
    ef_construction=512,
    m=16
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
