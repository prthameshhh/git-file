import streamlit as st
from langchain_groq import ChatGroq
from huggingface_hub import login
import langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import StdOutCallbackHandler
from langchain.cache import InMemoryCache
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
import os
import textwrap
import warnings
import time

warnings.filterwarnings("ignore")

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key='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'
)

# Prompt Template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

handler = StdOutCallbackHandler()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Streamlit UI
st.title("Gita AI using Mistral 7b")

# Get FAISS path dynamically
faiss_path = st.text_input("Enter the FAISS Database Path:", value="./db_faiss")

if not os.path.exists(faiss_path):
    st.error("Invalid FAISS Database Path. Please provide a valid path.")
else:
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        verbose=True,
        callbacks=[handler],
        chain_type_kwargs={'prompt': prompt}
    )

    def wrap_text_preserve_newlines(text, width=200):
        lines = text.split('\n')
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

    def process_llm_response(llm_response):
        ans = wrap_text_preserve_newlines(llm_response['result'])
        sources_used = ' \n'.join([os.path.basename(source.metadata['source']) for source in llm_response['source_documents']])
        ans = ans + '\n\nSources: \n' + sources_used
        return ans

    query = st.text_input("Ask a question about the Bhagavad Gita:")
    if st.button("Get Answer") and query:
        start = time.time()
        qa_response = qa_chain({"query": query})
        llm_response = process_llm_response(qa_response)
        end = time.time()
        time_elapsed = int(round(end - start, 0))
        
        st.text_area("Response", value=llm_response, height=300)
        st.write(f"Time elapsed: {time_elapsed} seconds")
