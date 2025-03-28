import json
import pandas as pd
import os

def create_directory_struct():
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/analysis/'):
        os.makedirs('./datasets/analysis/')

def get_filenames(directory_path: str) -> list[str]:
  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path) and item[0] != ".":
      files.append(item_path)
    elif os.path.isdir(item_path):
        files += get_filenames(directory_path + "/" + item) 
  return files


def clean_results(input_dict: dict, source: str) -> dict:
    clean_query_results = {}

    for i in range(0, len(input_dict['test_results'])):
        clean_record = {}

        clean_record['success'] = input_dict['test_results'][i]['success']
        clean_record['input'] = input_dict['test_results'][i]['input']
        clean_record['actual_output'] = input_dict['test_results'][i]['actual_output']
        clean_record['expected_output'] = input_dict['test_results'][i]['expected_output']
        clean_record['context'] = input_dict['test_results'][i]['context']
        clean_record['retrieval_context'] = input_dict['test_results'][i]['retrieval_context']
        clean_record['source'] = source

        metrics = input_dict['test_results'][i]['metrics_data']
        metric_record = {}
        for j in range(0, len(metrics)):
            metric_details = {}

            metric_details['threshold'] = metrics[j]['threshold']
            metric_details['success'] = metrics[j]['success']
            metric_details['score'] = metrics[j]['score']
            metric_details['reason'] = metrics[j]['reason']

            metric_record[metrics[j]['name']] = metric_details

        clean_record['metrics'] = metric_record
        clean_query_results[input_dict['test_results'][i]['name']] = clean_record
    return clean_query_results




def create_results_table(results: list, labels: list) -> pd.DataFrame:
    df_dict = {}
    df_dict['method'] = []
    df_dict['test_case'] = []
    df_dict['input'] = []
    df_dict['actual_output'] = []
    df_dict['expected_output'] = []
    df_dict['context'] = []
    df_dict['retrieval_context'] = []
    df_dict['source'] = []
    df_dict['answer_relevancy_score'] = []
    df_dict['answer_relevancy_reason'] = []
    df_dict['faithfulness_score'] = []
    df_dict['faithfulness_reason'] = []
    df_dict['contextual_relevancy_score'] = []
    df_dict['contextual_relevancy_reason'] = []

    for index, res in enumerate(results):
        for i in res.keys():
            df_dict['test_case'].append(i)
            df_dict['method'].append(labels[index])
            df_dict['input'].append(res[i]['input'])
            df_dict['actual_output'].append(res[i]['actual_output'])
            df_dict['expected_output'].append(res[i]['expected_output'])
            df_dict['context'].append(res[i]['context'])
            df_dict['retrieval_context'].append(res[i]['retrieval_context'])
            df_dict['source'].append(res[i]['source'])

            for j in res[i]['metrics'].keys():
                if j == 'Answer Relevancy':
                    df_dict['answer_relevancy_score'].append(res[i]['metrics'][j]['score'])
                    df_dict['answer_relevancy_reason'].append(res[i]['metrics'][j]['reason'])
                elif j == 'Faithfulness':
                    df_dict['faithfulness_score'].append(res[i]['metrics'][j]['score'])
                    df_dict['faithfulness_reason'].append(res[i]['metrics'][j]['reason'])
                elif j == 'Contextual Relevancy':
                    df_dict['contextual_relevancy_score'].append(res[i]['metrics'][j]['score'])
                    df_dict['contextual_relevancy_reason'].append(res[i]['metrics'][j]['reason'])
    return pd.DataFrame.from_dict(df_dict)

def loop_evals(res_filename: str, prefix:str="") -> None:
    create_directory_struct()
    files = get_filenames("./datasets/evals/")
    final_df = pd.DataFrame()

    for i, filename in enumerate(files):
        if prefix != filename.split('/')[-1][0:len(prefix)]:
            continue

        with open(filename, "r") as file:
            eval_json = json.load(file)
        source = filename.split('/')[-1].split('_')[2]

        record_json = clean_results(eval_json, source)
        df = create_results_table([record_json], [prefix.split('_')[0]])
        if i == 0:
            final_df = df
        if i > 0:
            final_df = pd.concat([final_df, df])

    final_df.to_csv("./datasets/analysis/" + res_filename + ".csv")
    
# loop_evals('base_results', 'base_')
# loop_evals('optimized_results', "optimized_")
# loop_evals('nosum_results', 'nosum_')
# loop_evals("noreranknosum_results", "noreranknosum_")
# loop_evals("noreranksum_results", "noreranksum_")
# loop_evals("denseopt_results", "denseopt_")
# loop_evals("densenosum_results", "densenosum_")
loop_evals("densenorerank_results", "densenorerank_")
