from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval import evaluate
from dotenv import load_dotenv
import json
import os

load_dotenv()

def create_directory_struct():
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/evals/'):
        os.makedirs('./datasets/evals/')

def eval_json(filename: str) -> None:
    if not os.path.exists(filename):
        print(f'{filename} does not exist in responses')
        return
    with open(filename, 'r') as responses_json:
        responses = json.load(responses_json)
    test_cases = []

    for index in responses.keys():
        test_case = LLMTestCase(
            input=responses[index]['input'],
            actual_output=responses[index]['actual_output'],
            expected_output=responses[index]['expected_output'],
            context=[str(responses[index]['context'])],
            retrieval_context=[str(responses[index]['retrieval_context'])],
        ) 
        test_cases.append(test_case)

    answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model='gpt-4o-mini')
    faithfulness = FaithfulnessMetric(threshold=0.5, model='gpt-4o-mini')
    contextual_relevancy = ContextualRelevancyMetric(threshold=0.5, model='gpt-4o-mini')

    evaluation = evaluate(test_cases, [answer_relevancy, faithfulness, contextual_relevancy],max_concurrent=6, ignore_errors=True, run_async=True,
                          throttle_value=5, use_cache=True, print_results=False) 
    with open('./datasets/evals/' + filename.split('/')[-1], "w") as f:
         json.dump(evaluation.model_dump(), f)

def get_filenames(directory_path: str) -> list[str]:
  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path) and item[0] != ".":
      files.append(item_path)
    elif os.path.isdir(item_path):
        files += get_filenames(directory_path + "/" + item) 
  return files

def loop_responses(prefix:str="") -> None:
    create_directory_struct()

    evaluated = set(get_filenames('./datasets/evals/'))

    for filename in get_filenames('./datasets/responses/'):
        if prefix and prefix not in filename.split('/')[-1][0:len(prefix)]:
            continue

        if './datasets/evals/' + filename.split('/')[-1] in evaluated:
            print(filename.split('/')[-1] + ' already evaluated evaluated; results cached')
            continue

        try:
            print(f'Evaluating ' + filename.split('/')[-1])
            eval_json(filename)
        except Exception as e:
            print(f'{filename} unable to be evaluated: {e}')

# loop_responses("base_")
# loop_responses("optimized_")
# loop_responses("nosum_")
loop_responses("noreranknosum_")
loop_responses("noreranksum_")
loop_responses("denseopt_")
loop_responses("densenosum_")
loop_responses("densenoreranknosum_")
