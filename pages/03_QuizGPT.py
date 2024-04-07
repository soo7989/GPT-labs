import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

quiz_format_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(docs):
    return "\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.

            Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the 
            text.

            Each question should have 4 answers, three of them must be incorrect and one should be correct.

            Context: {context}
            """,
        )
    ]
)


@st.cache_data(show_spinner="loading file")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="Making Quiz")
def run_quiz_chain(_docs, topic):
    chain = {"context": format_docs} | prompt | llm
    return chain.invoke(_docs)


with st.sidebar:
    docs = None
    topic = None

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

    choice = st.selectbox(
        "choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["docx", "txt", "pdf"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            docs = wiki_search(topic)


llm = ChatOpenAI(
    temperature=0.1,
    model=("gpt-3.5-turbo-1106"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=api_key,
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        quiz_format_function,
    ],
)


if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.
                
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)
    print(response)
    with st.form("questions_form"):
        for question in response["questions"]:
            value = st.radio(
                question["question"],
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")

        button = st.form_submit_button()
