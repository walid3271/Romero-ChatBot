import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text():

    urls = [
    "https://romero.sparktechwp.com//",
    "https://romero.sparktechwp.com/recommand-number-of-session/",
    "https://romero.sparktechwp.com/how-to-prepare-for-your-therapy-session/",
    "https://romero.sparktechwp.com/services/"
    ]

    all_text_documents = []

    for url in urls:
        loader = WebBaseLoader(web_paths=(url,), bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                class_=("elementor-widget-wrap elementor-element-populated",
                        "elementor-widget-container",
                        "elementor-heading-title elementor-size-default",
                        "elementor-element elementor-element-769046b elementor-widget elementor-widget-text-editor",
                        "elementor-element elementor-element-45caf6b elementor-widget elementor-widget-text-editor",
                        "elementor-element elementor-element-1d00220 e-con-full e-flex e-con e-child",
                        "elementor-element elementor-element-5b5f026 e-con-full e-flex e-con e-child",
                        "elementor-element elementor-element-8c9d408 e-con-full e-flex e-con e-child",
                        "elementor-element elementor-element-bbe4492 e-con-full e-flex e-con e-child",
                        "elementor-element elementor-element-bc9d464 e-con-full e-flex e-con e-child",
                        "elementor-background-overlay",
                        "elementor-element elementor-element-0ba78b3 e-flex e-con-boxed e-con e-parent e-lazyloaded",
                        "elementor-column elementor-col-100 elementor-top-column elementor-element elementor-element-f3c9b0b",
                        "elementor-column elementor-col-100 elementor-top-column elementor-element elementor-element-19aecd5",
                        "elementor-column elementor-col-50 elementor-inner-column elementor-element elementor-element-b7e5a68",
                        "elementor-column elementor-col-100 elementor-top-column elementor-element elementor-element-f40d52a",
                        "elementor-column elementor-col-100 elementor-top-column elementor-element elementor-element-e559b2b",
                        "elementor-column elementor-col-100 elementor-top-column elementor-element elementor-element-3a2e800",
                        "elementor-column elementor-col-100 elementor-top-column elementor-element elementor-element-a75bc91",
                        "elementor-column elementor-col-50 elementor-top-column elementor-element elementor-element-f91e1c7")
            ))
        )
        text_documents = loader.load()
        all_text_documents.extend(text_documents)
    
    return all_text_documents


# def get_text():
#     loader = WebBaseLoader(
#         web_paths=("https://romero.sparktechwp.com/recommand-number-of-session/",),
#         bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#             class_="elementor-element elementor-element-0573a8e e-flex e-con-boxed e-con e-child"
#         ))
#     )
#     text_documents = loader.load()
#     return text_documents

def get_text_chunks(text_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    documents = text_splitter.split_documents(text_documents)
    texts = [doc.page_content for doc in documents]
    return texts

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "answer is not available in the context" and do not provide an incorrect answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="ROMERO")
    st.header("ChatBot For ROMERO")

    user_question = st.text_input("Ask Questions About ROMERO")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":

    raw_text = get_text()
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    main()

# streamlit run bdCalling.py
