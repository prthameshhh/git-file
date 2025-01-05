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

# Initialize the language model and the embeddings
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key='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'
)

DB_FAISS_PATH = 'db_faiss'

# Custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Initialize the prompt template
prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

handler = StdOutCallbackHandler()

# Set up the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load the FAISS vector database
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    verbose=True,
    callbacks=[handler],
    chain_type_kwargs={'prompt': prompt}
)

# Helper functions for formatting and processing the response
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

# Function to handle the prediction
def predict(message):
    start = time.time()
    qa_response = qa_chain({"query": message})
    llm_response = process_llm_response(qa_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return llm_response + time_elapsed_str

# Streamlit Interface
def main():
    # Set title and description
    st.title("Gita AI using Mistral 7b")
    st.write("Ask anything related to Bhagavad Gita and get an answer.")

    # Provide examples for the user
    st.sidebar.title("Examples")
    examples = [
        "Can you explain briefly what the Bhagavad Gita teaches about life and spirituality?",
        "In a sentence, summarize the main message of the Bhagavad Gita.",
        "Explain the plot of the Bhagavad Gita in a sentence.",
        "How many chapters are there in the Bhagavad Gita, and what is the significance of each?",
        "Write a 100-word article on 'Impact of Bhagavad Gita on Personal Development.'",
    ]
    for example in examples:
        st.sidebar.button(example, on_click=lambda msg=example: st.session_state.user_input.update(msg))

    # Textbox for user input
    user_input = st.text_input("Ask a question:", key="user_input")

    # Handle user input and display the answer
    if user_input:
        with st.spinner("Generating answer..."):
            answer = predict(user_input)
            st.write(answer)

if __name__ == "__main__":
    main()
