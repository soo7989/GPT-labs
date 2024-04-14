from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import streamlit as st

st.set_page_config(
    page_title="Document GPT",
    page_icon="📚",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource(show_spinner="Embedding file")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("Document GPT")
st.markdown(
    """
    챗봇을 이용하여 문서의 내용을 정리할 수 있어요! 

    1. 사이드바에서 OPENAI_API_KEY 를 입력합니다. 
    
    2. 사이드바에서 문서를 업로드 합니다.
    
    3. 아래의 입력창을 통해 문서에서 필요한 내용을 질문합니다.   
    """
)

with st.sidebar:
    st.markdown(
        """
        GitHub: https://github.com/soo7989/GPT-labs

        """
    )

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key = st.text_input(
        "OPENAI_API_KEY",
        value=st.session_state["api_key"],
        key="api_key_input",
    )
    st.session_state["api_key"] = api_key

    file = st.file_uploader(
        "Upload your txt, pdf, docx",
        type=["pdf", "docx", "txt"],
    )


try:
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key,
    )
    print(api_key)
except Exception as e:
    print(e)
    print(api_key)

if file:
    retriever = embed_file(file)
    send_message("무엇을 할까요?", "ai", save=False)
    paint_history()
    message = st.chat_input("무엇을 할까요?")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
    else:
        st.session_state["messages"] = []
