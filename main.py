from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import gradio as gr

def process_file(file):
    text_chunks = []
   
    reader = PdfReader(file.name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
   
    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=30000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    text_chunks.extend(chunks)
   
    # Generate embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
   
    # Create vector store with Chroma
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    return vector_store

def answer_question(file, user_question):
    vector_store = process_file(file)
    # Perform similarity search
    match = vector_store.similarity_search(user_question)
   
    # Define the language model
    llm = ChatOllama(
        temperature=0,
        max_tokens=4000,
        model_name="ollama-3.1",
        model="ollama-3.1"  # Ensure this field is included
    )
   
    # Load QA chain and get response
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.invoke(input_documents=match, input=user_question)  # Corrected argument name
   
    return response

iface = gr.Interface(
    fn=answer_question,
    inputs=[gr.File(label="Upload PDF File"), gr.Textbox(lines=2, placeholder="Enter your question here...")],
    outputs="text",
    title="PDF Question Answering",
    description="Upload a PDF file and ask questions based on its content."
)

# Launch the interface
iface.launch()
