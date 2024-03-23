import streamlit as st
st.title("Restaurant Name Generator")
cuisine=st.sidebar.selectbox("Pick a Cuisine",("Indian","Italian","Mexican","Arabic"))
import LangChain

if cuisine:
    response=LangChain.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'])
    menu_items=response['menu_items'].strip().split(",")
    st.write('**Menu Items**')
    for item in menu_items:
        st.write(item)
