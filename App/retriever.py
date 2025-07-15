import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from App.config.settings import settings

def load_and_split_documents(docs_folder: str = "docs"):
   all_docs = []

   for filename in os.listdir(docs_folder):
      if filename.endswith(".txt"):
         filepath = os.path.join(docs_folder, filename)
         loader = TextLoader(filepath, encoding="utf-8")
         documents = loader.load()
         all_docs.extend(documents)

   splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   return splitter.split_documents(all_docs)

def create_retriever():
   print("Loading documents and creating retriever..")

   documents = load_and_split_documents()

   embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

   vectorstore = Chroma.from_documents(
      documents,
      embedding=embeddings,
      persist_directory="chroma_db"
   )

   print("Retriever ready.")
   return vectorstore.as_retriever(search_kwargs={"k": 4})