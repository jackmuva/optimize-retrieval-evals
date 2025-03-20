import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.openapi_support import PineconeApiException
from langchain_openai.chat_models import ChatOpenAI
from pinecone_text import sparse
load_dotenv()

def get_pinecone_indices(pc):
    try:
        dense = pc.Index(str(os.getenv('PINECONE_DENSE_INDEX')))
        sparse = pc.Index(str(os.getenv('PINECONE_SPARSE_INDEX')))
    except PineconeApiException:
        if not pc.has_index(str(os.getenv('PINECONE_DENSE_INDEX'))):
            pc.create_index_for_model(
                name=str(os.getenv('PINECONE_DENSE_INDEX')),
                cloud="aws",
                region="us-east-1",
                embed={ # pyright: ignore
                    "model":"multilingual-e5-large",
                    "field_map":{"text": "chunk_text"}
                }
            )
        dense = pc.Index(str(os.getenv('PINECONE_DENSE_INDEX')))

        if not pc.has_index(str(os.getenv('PINECONE_SPARSE_INDEX'))):
            pc.create_index_for_model(
                name=str(os.getenv('PINECONE_SPARSE_INDEX')),
                cloud="aws",
                region="us-east-1",
                embed={ # pyright: ignore
                    "model":"pinecone-sparse-english-v0",
                    "field_map":{"text": "chunk_text"}
                }
            )
        sparse = pc.Index(str(os.getenv('PINECONE_SPARSE_INDEX')))

    return dense, sparse 



def vector_search(index, query: str,top_k: int = 5, rerank:bool=False):
    if rerank:
        query_response = index.search_records(
            namespace=str(os.getenv('PINECONE_NAMESPACE')),
            query={
                "inputs": {"text": query},
                "top_k": top_k
            },
            rerank={
                "model": "cohere-rerank-3.5",
                    "rank_fields": ["text"]
            }
        )
    else:
        query_response = index.search_records(
            namespace=str(os.getenv('PINECONE_NAMESPACE')),
            query={
                "inputs": {"text": query},
                "top_k": top_k
            }
        )
    return query_response 

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


# def sparse_search(pc, index, query: str, top_k: int = 5):
#     dense_vector, sparse_vector = get_query_embeddings(pc, query)
#     print(sparse_vector)
#     query_response = index.query(
#         namespace=str(os.getenv('PINECONE_NAMESPACE')),
#         top_k=top_k,
#         sparseVector={"indices": sparse_vector['sparse_indices'], "values": sparse_vector['sparse_values']},
#         includeMetaData=True,
#         fields=["text", "source"]
#     )
#     return query_response 

def merge_chunks(dense_matches, sparse_matches):
    deduped_hits = {hit['_id']: hit for hit in dense_matches['result']['hits'] + sparse_matches['result']['hits']}.values()
    return sorted(deduped_hits, key=lambda x: x['_score'], reverse=True)

def run_rag(prompt: str, search_method:str="dense", top_k:int=5, rerank:bool=False):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    dense_index, sparse_index = get_pinecone_indices(pc)

    llm = ChatOpenAI(temperature=0.25, model="gpt-4o-mini") 
    messages = [
        (
            "system",
            '''You are a helpful assistant with access to an external knowledge base. 
                You will be given additional context in many of your interactions.''',
        ),
        ("human", prompt),
    ]
    
    if search_method == 'dense':
        response = vector_search(dense_index, prompt, top_k, rerank)
        text_answer = " ".join([doc['fields']['text'] for doc in response.result['hits']])
    elif search_method == 'sparse':
        response = vector_search(sparse_index, prompt, top_k, rerank)
        text_answer = " ".join([doc['fields']['text'] for doc in response.result['hits']])
    else:
        query_response = vector_search(dense_index, prompt, top_k, rerank)
        sparse_response = vector_search(sparse_index, prompt, top_k, rerank)
        response = {"result": {"hits": merge_chunks(query_response, sparse_response)}}
        text_answer = " ".join([doc['fields']['text'] for doc in response['result']['hits']])

    added_prompt = f"{text_answer} Using the provided information, give me a better and summarized answer"
    messages.append(("assistant", added_prompt))
    better_answer = llm.invoke(messages)
    return better_answer

# Call the function
final_answer = run_rag(prompt="what is paragon", search_method='hybrid', rerank=True)
print(final_answer)
