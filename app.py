import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
st.markdown("""
<style>
    /* Change the background color */
    .stApp {
        background-color: #F0F2F6;
    }
    /* Customize the title */
    h1 {
        color: #333;
        font-family: 'Garamond';
    }
    /* Customize subheader */
    .stSubheader {
        color: #555;
        font-family: 'Arial';
    }
    /* Customize button */
    .stButton>button {
        border: 2px solid #000000; /* Black border */
        border-radius: 20px;
        color: white; /* White text */
        background-color: #000000; /* Black background */
        padding: 10px 24px;
        cursor: pointer;
        font-size: 18px;
    }
    /* Customize text input */
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    /* Warning message style */
    .stAlert {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

with open("keys/.gemini_api_key.txt", "r") as f:
    GEMINI_API_KEY = f.read().strip()
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, 
                                               model="models/embedding-001")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Define retrieval function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from the user. Your answer should be based on the specific context."),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:")
])

chat_model = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, 
                                    model="gemini-1.5-pro-latest")
output_parser = StrOutputParser()
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)
st.image('inno.png', caption='Innovation Image')
st.title("Q&A RAG Chatbot")
st.subheader("A Retrieval-Augmented Generation System on the 'Leave No Context Behind' Paper")
question = st.text_input("Enter your question:", placeholder="Type your question here...")

if st.button("Ask"):
    if question:
        response = rag_chain.invoke(question)
        st.write("Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")