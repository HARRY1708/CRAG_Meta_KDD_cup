import bz2
import json
import os
from datetime import datetime
import argparse
import glob

from loguru import logger
from tqdm.auto import tqdm

import vllm
from openai import OpenAI



def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        if "true" in resp:
            answer = 1

        return answer
    except:
        return -1

# def evaluate_predictions_combined(results, eval_model):
#     """
#     Evaluate predictions for overall, domain, question_type, and static_or_dynamic in a single pass.

#     Args:
#         results (dict): The dataset containing queries, predictions, and ground truths.
#         eval_model (object): The evaluation model used for assessing predictions.

#     Returns:
#         tuple: Overall results, detailed evaluation results, and logs.
#     """
#     fields = ["domain", "question_type", "static_or_dynamic"]

#     # Initialize containers for detailed results and logs
#     detailed_evaluation_results = {field: {} for field in fields}
#     detailed_evaluation_logs = {field: {} for field in fields}

#     # Initialize overall metrics
#     overall_n_miss, overall_n_correct, overall_n_correct_exact, overall_n = 0, 0, 0, 0

#     # Prepare data structures for field-specific evaluations
#     for field in fields:
#         for key in results[field].keys():
#             detailed_evaluation_results[field][key] = {"evaluation_results": {}}
#             detailed_evaluation_logs[field][key] = {"evaluation_logs": []}
#             n_miss, n_correct, n_correct_exact = 0, 0, 0
#             field_queries, field_ground_truths, field_predictions = results[field][key]["queries"], results[field][key]["ground_truths"], results[field][key]["predictions"]
            
#             llm_evaluation_logs = [] # record queries that need llm evaluation

#             for _idx, prediction in tqdm(enumerate(field_predictions), total=len(field_predictions)):
#                 query = field_queries[_idx]
#                 ground_truth = str(field_ground_truths[_idx]).strip()
#                 prediction = prediction.strip()

#                 ground_truth_lowercase = ground_truth.lower()
#                 prediction_lowercase = prediction.lower()

#                 if "i don't know" in prediction_lowercase:
#                     n_miss += 1
#                     overall_n_miss += 1
#                     continue
#                 elif prediction_lowercase == ground_truth_lowercase:
#                     n_correct_exact += 1
#                     overall_n_correct_exact += 1
#                     n_correct += 1
#                     overall_n_correct += 1
#                     continue
#                 # else use llm evaluator to eval
#                 response = eval_model.evaluate(query, ground_truth, prediction)
#                 detailed_evaluation_logs[field][key]["evaluation_logs"].append({"query": query, "ground_truth": ground_truth, "prediction": prediction, "response": response})
#                 eval_res = parse_response(response)
#                 if eval_res == 1:
#                     n_correct += 1
#                     overall_n_correct += 1
                    
#             n = len(field_predictions)
#             print(f"{field}-n: ", n)
#             overall_n += n
#             detailed_evaluation_results[field][key]["evaluation_results"] = {
#                 "score": (2 * n_correct + n_miss) / n - 1,
#                 "exact_accuracy": n_correct_exact / n,
#                 "accuracy": n_correct / n,
#                 "hallucination": (n - n_correct - n_miss) / n,
#                 "missing": n_miss / n,
#                 "n_miss": n_miss,
#                 "n_correct": n_correct,
#                 "n_correct_exact": n_correct_exact,
#                 "total": n,
#             }
#     print("overall_n: ", overall_n)
#     overall_evaluation_results = {
#         "score": (2 * overall_n_correct + overall_n_miss) / overall_n - 1,
#         "exact_accuracy": overall_n_correct_exact / overall_n,
#         "accuracy": overall_n_correct / overall_n,
#         "hallucination": (overall_n - overall_n_correct - overall_n_miss) / overall_n,
#         "missing": overall_n_miss / overall_n,
#         "n_miss": overall_n_miss,
#         "n_correct": overall_n_correct,
#         "n_correct_exact": overall_n_correct_exact,
#         "total": overall_n,
#     }
#     return overall_evaluation_results, detailed_evaluation_results, detailed_evaluation_logs


# if __name__ == "__main__":
#     # ... (initial setup code remains unchanged)

#     # Load predictions
#     predictions_file = os.path.join(read_directory, "detailed_predictions.json")
#     results = json.load(open(predictions_file))

#     # Evaluate predictions
#     overall_evaluation_results, detailed_evaluation_results, detailed_evaluation_logs = evaluate_predictions_combined(results, eval_model)

#     # Save evaluation results
#     json.dump(detailed_evaluation_results, open(os.path.join(read_directory, "detailed_evaluation_results.json"), "w"), indent=4)
#     json.dump(detailed_evaluation_logs, open(os.path.join(read_directory, "detailed_evaluation_logs.json"), "w"), indent=4)
#     json.dump(overall_evaluation_results, open(os.path.join(read_directory, "evaluation_results.json"), "w"), indent=4)
#     json.dump(llm_evaluation_logs, open(os.path.join(read_directory, "llm_evaluation_logs.json"), "w"), indent=4)

    
def evaluate_predictions(results, eval_model):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    queries, ground_truths, predictions = results["queries"], results["ground_truths"], results["predictions"]

    llm_evaluation_logs = [] # record queries that need llm evaluation

    for _idx, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        query = queries[_idx]
        ground_truth = str(ground_truths[_idx]).strip()
        prediction = prediction.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        if "i don't know" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue
        # else use llm evaluator to eval
        response = eval_model.evaluate(query, ground_truth, prediction)
        llm_evaluation_logs.append({"query": query, "ground_truth": ground_truth, "prediction": prediction, "response": response})
        eval_res = parse_response(response)
        if eval_res == 1:
            n_correct += 1

    n = len(predictions) 
    
    evaluation_results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(evaluation_results)
    return evaluation_results, llm_evaluation_logs

def get_file_name(file_path):
    save_path = file_path.split("detailed_predictions_")[-1]
    return save_path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="data/crag_task_1_dev_v4_release.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2", # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2", # full data
                                 ])

    parser.add_argument("--model_name", type=str, default="vanilla_baseline",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 "modified_rag",
                                 "prompt_rag_bsln"
                                 ],
                        )
    parser.add_argument("--eval_llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--is_server", action="store_true", default=True,
                        help="Whether we use vLLM deployed on a server or offline inference.")
    parser.add_argument("--vllm_server", type=str, default="http://localhost:8088/v1",
                        help="URL of the vLLM server if is_server is True. The port number may vary.")
    parser.add_argument("--max_retries", type=int, default=10,
                        help="Number of retries for evaluation per query.")


    args = parser.parse_args()
    print(args.is_server)

    dataset_path = args.dataset_path            
    dataset = dataset_path.split("/")[0]            
    dataset_path = os.path.join("..", dataset_path)     

    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]

    # init evaluation model
    from evaluation_model import EvaluationModel
    eval_model = EvaluationModel(llm_name=llm_name, is_server=args.is_server,
                                 vllm_server=args.vllm_server, max_retries=args.max_retries)


    # get output directory
    model_name = args.model_name
    read_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
    output_directory = os.path.join("..", "output", dataset, model_name, args.llm_name)
    # if not os.path.exists(output_directory):
    #     raise FileNotFoundError(f"Output directory {output_directory} does not exist.")
    # print(read_directory, output_directory)
    
    print("this is the eval_mdoel : ", llm_name)
    print("this is the value of vllm_server: ", args.vllm_server)
    
    base_read_directory = "/home/jupyter/cs245-project-crag-master/final_predictions/"
    
    print(base_read_directory)
    
    prediction_file_paths = [i for i in glob.glob(base_read_directory + "*.json")]
    print("These are the predictions files to be evaluated: ", prediction_file_paths)
    # load predictions
    # predictions_file = os.path.join(read_directory, "detailed_predictions.json")
    
    for prediction_file_path in prediction_file_paths: 
        results = json.load(open(prediction_file_path))
        save_name = get_file_name(prediction_file_path)
        save_directory = "/home/jupyter/cs245-project-crag-master/final_results"
        save_directory = os.path.join(save_directory, save_name.split(".json")[0])
        print("this is the save directory: ", save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"{save_directory}: directory created!")

#         overall_evaluation_results, detailed_evaluation_results, detailed_evaluation_logs = evaluate_predictions_combined(results, eval_model)

#         evaluations = {"overall_evaluation_results": overall_evaluation_results, "detailed_evaluation_results": detailed_evaluation_results, "detailed_evaluation_logs": detailed_evaluation_logs}

#         json.dump(evaluations, open("cs245-project-crag-master/trial_predictions.json", "w"), indent=4)

        # Evaluate predictions
        overall_evaluation_results, overall_llm_evaluation_logs = evaluate_predictions(results, eval_model)

        domain_evaluation_results, static_or_dynamic_evaluation_results, question_type_evaluation_results = {}, {}, {}
        domain_llm_evaluation_logs, static_or_dynamic_llm_evaluation_logs, question_type_llm_evaluation_logs = {}, {}, {}

        fields = ['domain', 'question_type', "static_or_dynamic"]
        for field in fields:
            results_keys = {}
            logs_keys = {}
            for key in results[field].keys():
                results_keys[key] = {"evaluation_results": {}}
                logs_keys[key] = {"evaluation_logs": []}
            for key in results[field].keys():
                match field:
                    case 'domain':
                        domain_evaluation_results[field] = results_keys
                        domain_llm_evaluation_logs[field] = logs_keys
                    case 'question_type':
                        question_type_evaluation_results[field] = results_keys
                        question_type_llm_evaluation_logs[field] = logs_keys
                    case 'static_or_dynamic':
                        static_or_dynamic_evaluation_results[field] = results_keys
                        static_or_dynamic_llm_evaluation_logs[field] = logs_keys

        for field in fields:
            for key in results[field].keys():    # field specific keys ex. (domain: finance, movie, ...)
                evaluation_results, llm_evaluation_logs = evaluate_predictions(results[field][key], eval_model)
                match field:
                    case 'domain':
                        domain_evaluation_results['domain'][key]['evaluation_results'] = evaluation_results
                        domain_llm_evaluation_logs['domain'][key]['evaluation_logs'] = llm_evaluation_logs
                    case 'question_type':
                        question_type_evaluation_results['question_type'][key]['evaluation_results'] = evaluation_results
                        question_type_llm_evaluation_logs['question_type'][key]['evaluation_logs'] = llm_evaluation_logs
                    case 'static_or_dynamic':
                        static_or_dynamic_evaluation_results['static_or_dynamic'][key]['evaluation_results'] = evaluation_results
                        static_or_dynamic_llm_evaluation_logs['static_or_dynamic'][key]['evaluation_logs'] = llm_evaluation_logs
                    case default: 
                        print("Wrong Field!")

        detailed_evaluation_results = {'domain': domain_evaluation_results, 'static_or_dynamic': static_or_dynamic_evaluation_results, 'question_type': question_type_evaluation_results}
        detailed_evaluation_logs = {'domain': domain_llm_evaluation_logs, 'static_or_dynamic': static_or_dynamic_llm_evaluation_logs, 'question_type': question_type_llm_evaluation_logs}
        # save evaluation_results
        json.dump(detailed_evaluation_results, open(os.path.join(save_directory, "detailed_evaluation_results_" + save_name), "w"), indent=4)
        json.dump(detailed_evaluation_logs, open(os.path.join(save_directory, "detailed_evaluation_logs_" + save_name), "w"), indent=4)
        json.dump(overall_evaluation_results, open(os.path.join(save_directory, "evaluation_results" + save_name), "w"), indent=4)
        json.dump(overall_llm_evaluation_logs, open(os.path.join(save_directory, "llm_evaluation_logs" + save_name), "w"), indent=4)
