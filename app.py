import os
import csv
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import FileResponse 
from pydantic import BaseModel
from rag_engine import process_pdfs, get_agricultural_advice
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


os.environ["GOOGLE_API_KEY"] = "AIzaSyBQFspR1Jvb0qrX7caoAWt-jB2mE5_X2bU" 

app = FastAPI()

class WeatherData(BaseModel):
    condition: str

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

if os.path.exists("./chroma_db"):
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vector_db = process_pdfs("pdfs")

def log_to_csv(condition, advice):
    file_exists = os.path.isfile('farm_logs.csv')
    with open('farm_logs.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['التاريخ والوقت', 'سؤال الفلاح والطقس', 'رد الذكاء الاصطناعي']) 
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), condition, advice])

def log_unanswered(condition):
    file_exists = os.path.isfile('unanswered_logs.csv')
    with open('unanswered_logs.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['التاريخ والوقت', 'السؤال المفقود من المستندات']) 
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), condition])

@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.get("/analytics")
def read_analytics():
    return FileResponse("analytics.html")

@app.get("/missing-data")
def read_missing():
    return FileResponse("missing_data.html")

@app.get("/api/logs")
def get_logs():
    logs = []
    if os.path.exists("farm_logs.csv"):
        with open("farm_logs.csv", mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) 
            for row in reader:
                if len(row) >= 3:
                    logs.append({"date": row[0], "query": row[1], "reply": row[2]})
    return {"logs": logs[::-1]}

@app.get("/api/unanswered")
def get_unanswered_logs():
    logs = []
    if os.path.exists("unanswered_logs.csv"):
        with open("unanswered_logs.csv", mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) 
            for row in reader:
                if len(row) >= 2:
                    logs.append({"date": row[0], "query": row[1]})
    return {"logs": logs[::-1]}

@app.post("/get_advice")
def advice_endpoint(data: WeatherData):
    raw_advice = get_agricultural_advice(data.condition, vector_db)
    
  
    if "[NOT_IN_DOC]" in raw_advice:
       
        log_unanswered(data.condition)
       
        clean_advice = raw_advice.replace("[NOT_IN_DOC]", "").strip()
       
        log_to_csv(data.condition, clean_advice)
        return {"advice": clean_advice}
    else:
      
        log_to_csv(data.condition, raw_advice)
        return {"advice": raw_advice}