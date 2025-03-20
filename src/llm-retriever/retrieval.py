import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.openapi_support import PineconeApiException
from langchain_community.chat_models import ChatOpenAI
load_dotenv()

def get_pinecone_index(pc):
    try:
        index = pc.Index(str(os.getenv('PINECONE_INDEX')))
    except PineconeApiException:
        if not pc.has_index(str(os.getenv('PINECONE_INDEX'))):
            pc.create_index_for_model(
                name=str(os.getenv('PINECONE_INDEX')),
                cloud="aws",
                region="us-east-1",
                embed={ # pyright: ignore
                    "model":"multilingual-e5-large",
                    "field_map":{"text": "chunk_text"}
                }
            )
        index = pc.Index(str(os.getenv('PINECONE_INDEX')))
    return index

#
# def get_query_embeddings(pc, query: str) -> tuple[list, dict]:
#     dense_embeddings = pc.inference.embed(
#         model="multilingual-e5-large",
#         inputs=[query],
#         parameters={
#             "input_type": "passage"
#         }
#     )
#     sparse_embeddings = pc.inference.embed(
#         model="pinecone-sparse-english-v0",
#         inputs=[query],
#         parameters={"input_type": "passage", "return_tokens": True}
#     )
#     return dense_embeddings[0]['values'], sparse_embeddings[0]
#

def semantic_search(index, query: str,top_k: int = 5):
    query_response = index.search_records(
        namespace=str(os.getenv('PINECONE_NAMESPACE')),
        query={
            "inputs": {"text": query},
                "top_k": top_k
        }
    )
    return query_response 

def sparse_search(index, query: str, top_k: int = 5):
    sparse_vector = {} 
    query_response = index.query(
        namespace=str(os.getenv('PINECONE_NAMESPACE')),
        top_k=top_k,
        sparseVector=sparse_vector,
        includeMetaData=True
    )
    return query_response 

def run_rag(prompt: str):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = get_pinecone_index(pc)

    llm = ChatOpenAI(temperature=0.25, model="gpt-4o-mini") 
    messages = [
        (
            "system",
            '''You are a helpful assistant with access to an external knowledge base. 
                You will be given additional context in many of your interactions.''',
        ),
        ("human", prompt),
    ]
    query_response = semantic_search(index, prompt)
    text_answer = " ".join([doc['fields']['text'] for doc in query_response.result['hits']])
    added_prompt = f"{text_answer} Using the provided information, give me a better and summarized answer"
    messages.append(("assistant", added_prompt))
    better_answer = llm.invoke(messages)
    return better_answer

# Call the function
final_answer = run_rag(prompt="what is paragon")
print(final_answer)
