import streamlit as st
from PyPDF2 import PdfReader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain    # chat prompt
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv            # to load all the environments variable


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# function to read the PDF(pdf_docs) and convert the data into text 
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in PdfReader.pages:
            text += page.extract_text()
    return text

# divide the texts in chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20000)
    chunks=text_splitter.split_text(text)
    return chunks

# convert chunks into vectors
def get_vector_storage(text_chunks):     #store
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embeddings)   
    vector_store.save_local("faiss_index")  #faiss_index where the chunks get stored

# 
def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:

    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=PromptTemplate, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff", prompt=prompt)
    return chain

# user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)   # loading the text that is saved in faiss index
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain (
    {"input_documents":docs, "question": user_question},
     return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

# streamlit app
def main():
    st.set_page_config("PDF Reader")
    st.header("Chat with multiple PDFs using Gemini🤔")

    user_question = st.text_input("Search anything from the Thesis")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your thesis PDF files", accept_multiple_files= True)
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_storage(text_chunks)
                st.success("Done")

    
if __name__ == '__main__':
    main()