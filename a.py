import streamlit as st

import urllib.request
import os
import re
import openai

from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
os.environ["OPENAI_API_KEY"] = st.secrets["API"]


st.set_page_config(page_title="CHECK DETAILS FROM YOUR RESUME")
custom="""
<style>
div.css-1kyxreq.e115fcil2 
{
width: 50%;
margin: auto;
}


div.css-1v0mbdj.e115fcil1 > img
{
    width: 50%;
    margin-left: 23%;
}

div.block-container.css-1y4p8pa.ea3mdgi4
{
margin-top: -5%;
}


span.css-10trblm.e1nzilvr0
{
    margin: auto;
    padding-left: 23%;
    font-size: 32px;
    font-family: 'Poppins', sans-serif !important;
    line-height: 55px;
    letter-spacing: 3px;
}
div.css-5rimss.e1nzilvr4
{
    font-size: 15px;
    letter-spacing: 0.1px;
    line-height: 27px;
    font-family: "Poppins",sans-serif !important;
    font-weight: 400;
}
div.css-16idsys.e1nzilvr4   
{
    
    font-size: 15px;
    letter-spacing: 0.1px;
    line-height: 27px;
    font-family: "Poppins",sans-serif !important;
    font-weight: 400;
    margin-left: 34%;
}
div.st-b3.st-b8.st-bv.st-b1.st-bn.st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-bw.st-bs > input.st-bc.st-bx.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-b8.st-c7.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-ae.st-af.st-ag.st-cf.st-ai.st-aj.st-bw.st-cg.st-ch.st-ci
{
	font-size: 15px;
    letter-spacing: 0.1px;
    line-height: 27px;
    font-family: "Poppins",sans-serif !important;
    font-weight: 400;
        
}

div.st-bc.st-b3.st-bd.st-b8.st-be.st-bf.st-bg.st-bh.st-bi.st-bj.st-bk.st-bl.st-bm.st-b1.st-bn.st-au.st-ax.st-av.st-aw.st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-bo.st-bp.st-bq.st-br.st-bs.st-bt.st-bu
{
    border-radius: 30px;
    padding: 0 0 0 45px !important;
    border: 1px solid #3c7ed3;
    
}
<style>
"""

st.markdown(f"<style>{custom}</style>", unsafe_allow_html=True)



image_url = 'https://aposbook.com/static/new_blog_frontend/images/AposBook.png'
st.image(image_url)

st.header("KNOW ABOUT APOSBOOK")


# upload file
pdf = "sodapdf-converted-3.pdf"
    
    # extract the text
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
chunks = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)



      
user_question = st.text_input("What do you want to know about AposBook")
if user_question:
    docs = knowledge_base.similarity_search(user_question)
            
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
            

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
              
    st.write(response)

