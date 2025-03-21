import os
import json
from llm_retriever.retrieval import run_rag

def create_directory_struct():
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/responses/'):
        os.makedirs('./datasets/responses/')

def get_filenames(directory_path: str) -> list[str]:
  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path) and item[0] != ".":
      files.append(item_path)
    elif os.path.isdir(item_path):
        files += get_filenames(directory_path + "/" + item) 
  return files

def populate_response(response:dict, settings:dict={'search_method':'dense',
                                                    'top_k': 5,
                                                    'rerank': False,
                                                    'summarization': False,
                                                    'target_token': 500}) -> dict:
    result = response
    for row in result:
        answer, context = run_rag(result[row]['input'], search_method=settings['search_method'], top_k=settings['top_k'],
                                  rerank=settings['rerank'], summarization=settings['summarization'],
                                  target_token=settings['target_token'])
        result[row]['actual_output'] = answer.content
        result[row]['retrieval_context'] = context
    return result

def loop_prompts(search_method:str="dense", top_k:int=5, rerank:bool=False, summarization:bool=False, target_token=500):
    create_directory_struct()
    settings = {'search_method':search_method,
                'top_k': top_k,
                'rerank': rerank,
                'summarization': summarization,
                'target_token': target_token,
                }

    for filename in get_filenames("./datasets/json-goldens/"):
        if os.path.exists('./datasets/responses/' + filename.split('/')[-1]):
            print(f'cached result, skipping {filename}')
            continue
        
        response = {}
        if os.path.exists('./datasets/json-goldens/' + filename.split('/')[-1]):
            print("processing " + filename)
            try:
                with open(filename, 'r') as json_file:
                    response = json.load(json_file)
            except:
                print("empty file")
        else:
            print(filename + 'does not exist')
            continue
        
        try: 
            result_dict = populate_response(response, settings)
            with open('./datasets/responses/' + filename.split('/')[-1], 'w') as res_json_file:
                json.dump(result_dict, res_json_file)
            print(f'Success! Responsed to {filename}')
        except Exception as e:
            print(f'Unable to populate for {filename}: {e}')

loop_prompts()
