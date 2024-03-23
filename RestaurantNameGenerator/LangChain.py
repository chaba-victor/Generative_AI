from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import SequentialChain
import os
import streamlit as st
# Set the OpenAI API key
api_key = st.secrets['api_key']

llm=OpenAI(openai_api_key=api_key)
def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(input_variables=['cuisine'],
                                          template='I want to open restaurant for {cuisine} food. Suggest a single name.')
    prompt_template_name.format(cuisine="Italian")
    name_chain = LLMChain(prompt=prompt_template_name, llm=llm, output_key='restaurant_name')

    prompt_template_items = PromptTemplate(input_variables=['restaurant_name'],
                                           template='Suggest some menu items for {restaurant_name} food. Return it as a comma separator.')
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='menu_items')
    chain = SequentialChain(chains=[name_chain, food_items_chain],
                            input_variables=['cuisine'],
                            output_variables=['restaurant_name', 'menu_items'])
    response = chain.invoke({'cuisine': cuisine})
    return response

if __name__=="__main__":
    print(generate_restaurant_name_and_items("Italian"))