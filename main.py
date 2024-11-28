from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

## Retrieval
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



def upload_pdf(file_path):
    # Local PDF file uploads
    if local_path:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()
        return data
    else:
        print("Upload a PDF file")


def text_splitter(pdf_data):
    ## Initialize the text splitter
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=75)

    ## Split the extracted text into chunks
    chunks = txt_splitter.split_documents(pdf_data)

    return chunks


def vector_DB(chunks):
    ## Create vector DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="local-rag"
    )
    return vector_db

##########################################################

def part_1(local_path):

    pdf_data = upload_pdf(local_path)

    chunks = text_splitter(pdf_data)

    vector_db = vector_DB(chunks)

    return vector_db

##########################################################

def import_model(local_model):
    llm = ChatOllama(model=local_model)
    return llm



def prompt_template():

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    return QUERY_PROMPT

def multi_query_retriever(vector_db, llm, QUERY_PROMPT):

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    return retriever

def chat_prompt_template():
    
    ## RAG prompt
    template =   """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return prompt

def lang_chain(retriever, prompt, llm):

    chain= (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt| llm | StrOutputParser()
    )

    return chain

    

##########################################################
##########################################################

local_path = "rugby-injuries.pdf"

vector_db = part_1(local_path)
##########################################################

QUERY_PROMPT = prompt_template()

local_model = "llama3.2:1b"

llm = import_model(local_model)

retriever = multi_query_retriever(vector_db, llm, QUERY_PROMPT)

prompt = chat_prompt_template()

chain = lang_chain(retriever, prompt, llm)

# print(chain.invoke(input("? ")))


while True:
    # Get user input
    user_input = input("Ask your question: ")
    
    # Check for exit condition
    if user_input.strip().lower() == "exit":
        print("Exiting the system. Goodbye!")
        break

    # Process the user question through the chain
    try:
        response = chain.invoke({"question": user_input})
        print("\nAnswer:\n", response, "\n")
        
    except Exception as e:
        print("An error occurred:", str(e))