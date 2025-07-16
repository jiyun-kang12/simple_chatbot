##################################################################
#  streamlit/05_streamlit_chat_exam_session_state_llm_streaming_memory.py
##################################################################
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine




# 실제응답을 LLM에 요청(프롬프트 -> LLM 요청 -> 응답 -> chat_message container에 출력)

st.title("Chatbot+session state 튜토리얼")


# LLM 모델 생성
@st.cache_resource #이 함수가 한번만 실행될 수 있도록 cache에 저장.  (데코레이터 사용)
def get_llm_model():
    load_dotenv() # API 가져올 때 (.env에 설정해둔 apikey가 있을때)
    prompt_template = ChatPromptTemplate([
        ("system", "답변을 100단어 이내로 작성해주세요."),
        MessagesPlaceholder(variable_name="history", optional=True),  # ("placeholder", "{history}")
        ("human", "{query}")
    ])
    model = ChatOpenAI(model = "gpt-4o-mini")
    return prompt_template | model | StrOutputParser()

@st.cache_resource
def get_chain():
    # RunnableWithMessageHistory를 생성해서 반환.
    engine = create_engine("sqlite:///chat_history.sqlite")
    runnable = get_llm_model()
    chain = RunnableWithMessageHistory(
        runnable = runnable,
        get_session_history =lambda session_id : SQLChatMessageHistory(session_id=session_id, connection= engine),
        input_messages_key = "query",
        history_messages_key = "history"
    )

    return chain 

model = get_chain()


# Session State를 생성
## session_state: dictionary 구현체. 시작~종료할 때까지 사용자 별로 유지되어야 하는 값들을 저장하는 곳.

# 0. 대화 내역을 session state의 "messages": list로 저장
# 1. session state에 "messages" key가 있는지 조회(없으면 생성)
if "messages" not in st.session_state: 
    st.session_state["messages"] = []  # 대화내용들을 `저장할 리스트를 "messages" 키로 저장.  # 처음엔 빈리스트를 만들어둠.

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

# Sidebar에 session_id 입력 위젯 생성
session_id =st.sidebar.text_input("Session ID", placeholder = "대화 ID를 입력하세요.")


######################################
# 기존 대화 이력을 출력.
######################################
for message in st.session_state["messages"]:
    with st.chat_message(message['role']):
        st.write(message['content'])

# 사용자의 프롬프트(질문)을 입력받는 위젯
prompt = st.chat_input("User Prompt")  # 사용자가 입력한 문자열을 반환.


# session state에 저장할 수도 있음
# if "model" not in st.session_state:
#     st.session_state['model'] =  ChatOpenAI(model = "gpt-4o-mini")
# model = st.session_state['model']

# 대화 작업
if prompt is not None:
    # session_state에 messages에 대화내역을 저장.
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


    # 입력된 Session_id를 상태값으로 저장
    if st.session_state["session_id"] is None:
        st.session_state['session_id'] = session_id
    
    config = {"configurable": {"session_id":st.session_state['session_id']}}


    with st.chat_message("ai"):
        message_placeholder = st.empty()  #update가 가능한 container
        full_message = ""  # LLM이 응답하는 토큰들을 저장할 문자열변수.
        for token in model.stream({"query": prompt}, config=config):
            full_message += token
            message_placeholder.write(full_message)  # 기존 내용을 full_message로 갱신.
            #a -> ab -> abc 이런식으로 계속 full_message를 갱신하면 하나씩 나온느 것처럼 보임.
            print(full_message)

        st.session_state["messages"].append({"role": "ai", "content":full_message})     


