from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json
from langchain.text_splitter import MarkdownTextSplitter
from pinecone.openapi_support import PineconeApiException
from pinecone_text.sparse import BM25Encoder

load_dotenv()

def get_text_from_md(file_path: str) -> str | None:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_text = file.read()
        return markdown_text
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_filenames(directory_path: str) -> list:
  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path) and item[0] != ".":
      files.append(item_path)
    elif os.path.isdir(item_path):
        files += get_filenames(directory_path + "/" + item) 
  return files

def chunk_md(md: str,chunk_size=507, chunk_overlap=20) -> list:
    md_text = get_text_from_md(md)
    markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = markdown_splitter.create_documents([str(md_text)])

    chunks = []
    for i, doc in enumerate(docs):
        chunk_dict = {}
        chunk_dict['id'] = md.split("/")[-1] + f'/{i}'
        chunk_dict['text'] = doc.page_content
        chunk_dict['filename'] = md.split("/")[-1]
        chunks.append(chunk_dict)
    return chunks

# def train_bm25(chunk_size: int):
#     corpus = []
#
#     md_files = get_filenames('./knowledge-base')
#     for md in md_files:
#         corpus += chunk_md(md, chunk_size, 0)
#
#     bm25 = BM25Encoder()
#     bm25.fit(corpus)
#     return bm25

def upsert_chunk(chunks: list, pc, index, bm25 = None) -> None:
    #max sequences per batch: 96
    i = 0
    while i <= len(chunks):
        dense_embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in chunks[i:i+96]],
            parameters={
                "input_type": "passage"
            }
        )

        sparse_embeddings = pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=[d['text'] for d in chunks[i:i+96]],
            parameters={"input_type": "passage", "return_tokens": True}
        )

        vectors = []
        for data, dense, sparse in zip(chunks[i:i+96], dense_embeddings, sparse_embeddings):
            vectors.append({
                "id": data['id'],
                "values": dense['values'],
                "metadata": {'text': data['text'], 'source': data['filename']},
                "sparse_values": {'indices': sparse['sparse_indices'], 'values': sparse['sparse_values']}
            })
            
        index.upsert(
            vectors=vectors,
            namespace=str(os.getenv('PINECONE_NAMESPACE') or "")
        )
        i+=96

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

def clear_namespace():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if type(os.getenv('PINECONE_INDEX')) != type(None):
        index = get_pinecone_index(pc)
    else:
        print("Add a PINECONE_INDEX to .env file")
        return

    index.delete(delete_all=True, namespace=str(os.getenv('PINECONE_NAMESPACE') or ""))

def index_pinecone_with_knowledge_bases(chunk_size=512, chunk_overlap=20):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if type(os.getenv('PINECONE_INDEX')) != type(None):
        index = get_pinecone_index(pc)
    else:
        print("Add a PINECONE_INDEX to .env file")
        return

    if not os.path.exists('./cache/'):
        os.makedirs('./cache/')

    # bm25 = train_bm25(chunk_size)

    md_files = get_filenames('./knowledge-base')
    for md in md_files:
        chunks = chunk_md(md, chunk_size, chunk_overlap)
        try:
            index_cache = {}
            if os.path.exists('./cache/index-cache.json'):
                with open('./cache/index-cache.json', 'r') as file:
                    index_cache = json.load(file)
            if md.split('/')[-1] in index_cache:
                print(md.split('/')[-1] + " result cached; skipping")
                continue
            else:
                print(md.split('/')[-1] + " upserting")
                upsert_chunk(chunks, pc, index)
                index_cache[md.split('/')[-1]] = True
                with open('./cache/index-cache.json', 'w') as file:
                    json.dump(index_cache, file)
        except Exception as error:
            print(f'Unable to upsert document: {chunks[0]['filename']}')
            print(error)

index_pinecone_with_knowledge_bases()
