from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json
from langchain.text_splitter import MarkdownTextSplitter
from pinecone.openapi_support import PineconeApiException

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

def upsert_chunk(chunks: list, pc, index) -> None:
    #max sequences per batch: 96
    i = 0
    while i <= len(chunks):
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in chunks[i:i+96]],
            parameters={
                "input_type": "passage"
            }
        )

        vectors = []
        for d, e in zip(chunks[i:i+96], embeddings):
            vectors.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text'], 'source': d['filename']}
            })
            
        index.upsert(
            vectors=vectors,
            namespace=str(os.getenv('PINECONE_NAMESPACE') or "")
        )
        i+=96

def get_pinecone(pc):
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


def index_pinecone_with_knowledge_bases(chunk_size=512, chunk_overlap=20):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if type(os.getenv('PINECONE_INDEX')) != type(None):
        index = get_pinecone(pc)
    else:
        print("Add a PINECONE_INDEX to .env file")
        return

    if not os.path.exists('./cache/'):
        os.makedirs('./cache/')

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
