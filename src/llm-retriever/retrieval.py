from dataclasses import fields
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.openapi_support import PineconeApiException

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
def query_pinecone_index(index, query, top_k: int = 5, include_metadata: bool = True):
    query_response = index.search_records(
        namespace=str(os.getenv('PINECONE_NAMESPACE')),
        query={
            "inputs": {"text": query},
                "top_k": top_k
        }
    )
    return query_response


def semantic_search(query: str):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = get_pinecone_index(pc)

    # query_embeddings = get_query_embeddings(pc, query)
    answers = query_pinecone_index(index=index, query=query, top_k=5, include_metadata=True)
    return answers

print(semantic_search('what is Paragon'))
