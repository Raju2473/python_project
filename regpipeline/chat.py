from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI  # ✅ only this needed
from langchain_qdrant import QdrantVectorStore 




embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_db=QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="learning_RAG",

)


user_input=input("Ask something")

search_results=vector_db.similarity_search(query=user_input)

context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('page_label', 'N/A')}\n"
    f"File Location: {result.metadata.get('source', 'N/A')}"
    for result in search_results
])




SYSTEM_PROMPT=f"""
   you are a helpful AI Assistant who answeres user query based in the available context retrieved
   from a PDF file along with page_contents and page number.

   You should only answer the user based on the following context and navigate the user to open the right page
   number to know more.


   context:
   {context}
"""


response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":"system", "content":SYSTEM_PROMPT
            },{
                 "role":"user", "content": user_input
            }
        ]
    )
print(response.choices[0].message.content)