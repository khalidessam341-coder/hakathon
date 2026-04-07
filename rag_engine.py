import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


os.environ["GOOGLE_API_KEY"] = "AIzaSyC8j0QFoQLueIYC3b_mKdiqJsdA_u1pwYQ"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def process_pdfs(pdf_folder):
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    return vector_store

def get_agricultural_advice(weather_condition, vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )
    
    
    prompt = f"""
    أنت الآن "خبير زراعي رقمي" لمنصة نماء.
    المعطيات: طقس ({weather_condition}) + مستندات مرفقة.
    
    التعليمات الصارمة:
    1. ابحث في المستندات المرفقة أولاً.
    2. إذا وجدت المعلومة في المستندات، أجب مباشرة بشكل احترافي.
    3. إذا لم تجد المحصول أو المعلومة في المستندات المرفقة، يجب أن تبدأ إجابتك بالكود التالي تماماً: [NOT_IN_DOC]
    4. بعد الكود، قدم أفضل نصيحة زراعية عامة بناءً على خبرتك بخصوص حالة الطقس {weather_condition}، واذكر بوضوح أن هذه النصيحة عامة وليست من المستندات الرسمية الحالية.
    """
    
    response = qa_chain.invoke({"question": prompt})
    return response['answer']
