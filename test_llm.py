import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

for model_name in ["gemini-1.5-flash", "gemini-1.5-flash-latest"]:
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=1)
        res = llm.invoke("Hi")
        print(f"{model_name}: SUCCESS")
        break
    except Exception as e:
        print(f"{model_name}: ERROR - {str(e)}")
