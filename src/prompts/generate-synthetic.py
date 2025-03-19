from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Evolution
from dotenv import load_dotenv
import os
from markdown import Markdown
from io import StringIO
import json

load_dotenv()

def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
Markdown.output_formats["plain"] = unmark_element # pyright: ignore
__md = Markdown(output_format="plain")# pyright: ignore
__md.stripTopLevelTags = False


def unmark(text):
    return __md.convert(text)


def create_directory_struct():
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/txt-files/'):
        os.makedirs('./datasets/txt-files/')
    if not os.path.exists('./datasets/json-goldens/'):
        os.makedirs('./datasets/json-goldens/')


def get_filenames(directory_path: str) -> list[str]:
  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path) and item[0] != ".":
      files.append(item_path)
    elif os.path.isdir(item_path):
        files += get_filenames(directory_path + "/" + item) 
  return files

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

def convert_md_to_txt(filepath: str) -> None:
    txt_contents = unmark(get_text_from_md(filepath))   
    with open('./datasets/txt-files/' + filepath.split('/')[-1][:-3] + '.txt', 'w') as file:
        file.write(txt_contents)

def convert_all_md_to_txt() -> None:
    txt_set = set(get_filenames('./datasets/txt-files'))
    for filename in get_filenames('./knowledge-base/'):
        if filename.split('/')[-1][:-3] + '.txt' in txt_set:
            continue
        else:
            convert_md_to_txt(filename)

def generateGoldens():
    create_directory_struct()
    convert_all_md_to_txt()
    
    generated_set = set([x.split('/')[-1] for x in get_filenames('./datasets/json-goldens')])
    print(generated_set)
    for file in get_filenames('./datasets/txt-files/'):
        print(file.split('/')[-1][:-4] + ".json")
        if file.split('/')[-1][:-4] + ".json" in generated_set:
            print(f'skipping golden generation for {file}')
        else:
            print(f'generating goldens for {file}')
            try:
                synthesizer = Synthesizer(
                    evolution_config=EvolutionConfig(evolutions={
                        Evolution.REASONING: 1/7,
                        Evolution.MULTICONTEXT: 1/7,
                        Evolution.CONCRETIZING: 1/7,
                        Evolution.CONSTRAINED: 1/7,
                        Evolution.COMPARATIVE: 1/7,
                        Evolution.HYPOTHETICAL: 1/7,
                        Evolution.IN_BREADTH: 1/7,
                    },
                    num_evolutions=1),
                    model='gpt-4o-mini',
                )
                synthesizer.generate_goldens_from_docs(
                    document_paths=[file],
                    context_construction_config=ContextConstructionConfig(chunk_size=507, chunk_overlap=20, critic_model='gpt-4o-mini'),
                )
            
                golden_dataframe = synthesizer.to_pandas()
                golden_json = json.loads(str(golden_dataframe.to_json(orient='index')))

                result_json = {}
                for index in golden_json.keys():
                    result_json[file.split('/')[-1] + "-" + index] = golden_json[index]

                    with open('./datasets/json-goldens/' + file.split('/')[-1][:-4] + ".json", "w") as jsonFile:
                        json.dump(result_json, jsonFile)
            except:
                print(f'unable to generate goldens for {file}')

generateGoldens()
