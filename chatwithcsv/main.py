import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
uploaded_file=st.sidebar.file_uploader("Upload your data",type='csv')
api_key=st.secrets['api_key']
print(uploaded_file)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path=tmp_file.name
    agent = create_csv_agent(
        ChatOpenAI(temperature=0.5, openai_api_key=api_key),
        tmp_file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    def conversational_chat(query):
        response = agent.invoke(query)
        st.session_state['history'].append((query,response["output"]))
        return response['output']

    if 'history' not in st.session_state:
        st.session_state['history']=[]

    if 'generated' not in st.session_state:
        st.session_state['generated']=["Hello, Ask me anything about "+ uploaded_file.name]
    if 'past' not in st.session_state:
        st.session_state['past']=['Hey!']
    response_container=st.container()
    container=st.container()
    with container:
        with st.form(key="my_form",clear_on_submit=True):
            user_input=st.text_input('Query:',placeholder="Talk to your csv Data",key='input')
            submit_button=st.form_submit_button(label='chat')
            if submit_button and user_input:
                output=conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i],is_user=True,key=str(i)+'_users')
                message(st.session_state["generated"][i],key=str(i))
