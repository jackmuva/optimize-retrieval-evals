import os
from dotenv import load_dotenv
from huggingface_hub.inference._generated.types import summarization
from pinecone import Pinecone
from pinecone.openapi_support import PineconeApiException
from langchain_openai.chat_models import ChatOpenAI
from llmlingua import PromptCompressor

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

def compress_context(llm_lingua, context: list[str], instruction: str, question:str, target_token=500):
    compressed_prompt = llm_lingua.compress_prompt(
        context=context,
        instruction=instruction,
        question=question,
        target_token=target_token,
        rank_method="longllmlingua",
        dynamic_context_compression_ratio=0.4,  # enable dynamic_context_compression_ratio
        reorder_context="sort",
    )
    print(f"""savings: {compressed_prompt['saving']} | origin_tokens: {compressed_prompt['origin_tokens']} | compressed_tokens: {compressed_prompt['compressed_tokens']}""")
    return compressed_prompt['compressed_prompt']

    


pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
llm_lingua = PromptCompressor(model_name="openai-community/gpt2", device_map="cpu")
def run_rag(prompt: str, search_method:str="dense", top_k:int=5, rerank:bool=False, summarization:bool=False, target_token=500):
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
        context=[doc['fields']['text'] for doc in response.result['hits']]
        text_answer = " ".join(context)
    elif search_method == 'sparse':
        response = vector_search(sparse_index, prompt, top_k, rerank)
        context=[doc['fields']['text'] for doc in response.result['hits']]
        text_answer = " ".join(context)
    else:
        query_response = vector_search(dense_index, prompt, top_k, rerank)
        sparse_response = vector_search(sparse_index, prompt, top_k, rerank)
        response = {"result": {"hits": merge_chunks(query_response, sparse_response)}}
        context=[doc['fields']['text'] for doc in response['result']['hits']]
        text_answer = " ".join(context)

    instruction = "Using the provided information, give me a better and summarized answer"
    if summarization:
        added_prompt=compress_context(llm_lingua, context, instruction, prompt, target_token)
    else:
        added_prompt = f"{text_answer} {instruction}"

    messages.append(("assistant", added_prompt))
    better_answer = llm.invoke(messages)
    return better_answer, context 
