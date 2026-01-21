import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from lang_wrapper import MyCustomLLM
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = text_splitter.split_documents(data)
    return FAISS.from_documents(docs, embeddings)

db = load_vectorstore()

st.set_page_config(page_title="Custom LLM RAG Agent", page_icon="")
st.title(" Custom LLM RAG Agent")
st.markdown("Ask questions about your uploaded documents using a custom-trained LLM")



@st.cache_resource
def load_model():
    return MyCustomLLM("model/weights.pth", "model/model_config.json")


try:
    llm = load_model()
    st.sidebar.success(" Model loaded successfully!")
except Exception as e:
    st.error(f" Error loading model: {e}")
    st.stop()

st.sidebar.header(" Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a text file", type="txt")

if uploaded_file:

    raw_text = uploaded_file.read().decode("utf-8")

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    texts = text_splitter.split_text(raw_text)

    st.sidebar.info(f"Document split into {len(texts)} chunks")

    with st.spinner("Creating knowledge base..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.db = FAISS.from_texts(texts, embeddings)


    st.sidebar.success(" Knowledge base ready!")


    def ask_question(question, k=3):

        docs = st.session_state.db.similarity_search(question, k=k)


        context = "\n\n".join([f"Document {i + 1}:\n{doc.page_content}"
                               for i, doc in enumerate(docs)])


        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

        answer = llm(prompt)

        return answer, docs

  


    st.divider()


    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source_docs = ask_question(prompt)

            st.markdown(answer)

            with st.expander(" View Sources"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Source {i + 1}:**")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info(" Please upload a document to start asking questions")

    st.subheader(" Demo Mode")
    demo_prompt = st.text_input("Test the LLM directly (without RAG):")
    if demo_prompt:
        with st.spinner("Generating..."):
            response = llm(demo_prompt)
        st.write("**Response:**")
        st.write(response)
