import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

messages = [
    SystemMessagePromptTemplate.from_template(
        "You are a truthful, accurate AI agent that responds to the user's questions, given an AI paper by Apple."
    ),
    HumanMessagePromptTemplate.from_template("What is the paper about, in summary?"),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)
chat_model = ChatOllama(
    model="mistral",
)

loader = PyPDFLoader("./llm_in_a_flash_apple.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

print(pages[0])


db = None
if not os.path.exists("./faiss_index"):
    db = FAISS.from_documents(pages, embeddings)
    db.save_local("./faiss_index")
else:
    db = FAISS.load_local("faiss_index", embeddings)

query = "What is the paper about?"
docs = db.similarity_search_with_score(query)
print(docs[0])

ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 0.8}),
    verbose=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
)
