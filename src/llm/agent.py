import os

# from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("GROQ_API_KEY"))

# print("GROQ_API_KEY:", os.environ.get("GROQ_API_KEY"))

# llm = ChatGroq(
#     model="llama3-70b-8192",
#     api_key=os.environ["GROQ_API_KEY"],
# )
