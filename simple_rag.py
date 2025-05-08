import os
import ollama as o
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader

class Chatbot:
    def __init__(self, paths: list, embedding_model: str, llm: str):
        self.paths = paths
        o.pull(embedding_model)
        self.embedding_model = embedding_model
        self.llm = ChatOllama(model=llm)
        self.chain = self.create_rag()

    def create_rag(self):
        documents = []
        for path in self.paths:
            if os.path.exists(path):
                documents.extend(UnstructuredPDFLoader(path).load())
            else:
                print(f'{path} not found. Moving to the next document.')
        chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100).split_documents(documents)
        vector_db = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings(model=self.embedding_model), collection_name='internship_rag')
        template = "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}."
        query_prompt = PromptTemplate(input_variables=['question'], template=template)
        retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), self.llm, prompt=query_prompt)
        template = "Question: {question}\nAnswer above question based ONLY on the following context:\n{context}\nDo not give false information if the document doesn't contain anything regarding that topic"
        prompt = ChatPromptTemplate.from_template(template)
        chain = ({"question": RunnablePassthrough(), "context": retriever} | prompt | self.llm | StrOutputParser())
        return chain

    def answer(self, question):
        return self.chain.invoke(input=question)

def main():
    PDF_PATHS = ['2025 Helios Intern Handbook.pdf', 'Helios Training 2025.pdf']
    EMBEDDING_MODEL = 'nomic-embed-text'
    LLM = 'llama3.2:1b'
    chatbot = Chatbot(PDF_PATHS, EMBEDDING_MODEL, LLM)
    question = "What clothes are not allowed to be worn and why?"
    response = chatbot.answer(question)
    print(response)

if __name__ == '__main__':
    main()
