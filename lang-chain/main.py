from langchain.llms.openai import OpenAI
import streamlit as st
from config import config


st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic u want;")


llm = OpenAI(temperature=0.9)

if input_text:
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": input_text}
        ]
    )
    st.write(response.choices[0].message.content)
