import bz2
import json
import os
from datetime import datetime
import argparse

from loguru import logger
from tqdm.auto import tqdm


def load_data_in_batches(dataset_path, batch_size, split=-1):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": [], "domain": [], "static_or_dynamic": [], "question_type": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)

                    if split != -1 and item["split"] != split:
                        continue

                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e

def assign(field, value, batch, i, ground_truths, predictions):
    field[value]['queries'].append(batch['query'][i])
    field[value]['ground_truths'].append(ground_truths[i])
    field[value]['predictions'].append(predictions[i])
    
def generate_predictions(dataset_path, model, split):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    domain = {
        "finance": {"queries": [], "ground_truths": [], "predictions": []}, 
        "music": {"queries": [], "ground_truths": [], "predictions": []}, 
        "movie": {"queries": [], "ground_truths": [], "predictions": []}, 
        "sports": {"queries": [], "ground_truths": [], "predictions": []}, 
        "open": {"queries": [], "ground_truths": [], "predictions": []}
    }
    
    question_type = {
        "simple": {"queries": [], "ground_truths": [], "predictions": []}, 
        "simple_w_condition": {"queries": [], "ground_truths": [], "predictions": []}, 
        "comparison": {"queries": [], "ground_truths": [], "predictions": []}, 
        "aggregation": {"queries": [], "ground_truths": [], "predictions": []}, 
        "set": {"queries": [], "ground_truths": [], "predictions": []}, 
        "false_premise": {"queries": [], "ground_truths": [], "predictions": []}, 
        "post-processing": {"queries": [], "ground_truths": [], "predictions": []}, 
        "multi-hop": {"queries": [], "ground_truths": [], "predictions": []}
    }
    
    static_or_dynamic = {
        "static": {"queries": [], "ground_truths": [], "predictions": []}, 
        "slow-changing": {"queries": [], "ground_truths": [], "predictions": []}, 
        "fast-changing": {"queries": [], "ground_truths": [], "predictions": []}, 
        "real-time": {"queries": [], "ground_truths": [], "predictions": []}
    }
    
    queries, ground_truths, predictions = [], [], []
    batch_size = model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size, split), desc="Generating predictions"):
        # batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        # print("this is the value of batch_predictions: ", batch_ground_truths)
        # print("this is the type of batch_predictions: ", type(batch_ground_truths))
        # print("this is the value of domain['batch']: ", batch['static_or_dynamic'][0])
        # break
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions = model.batch_generate_answer(batch)
        
        for i in range(batch_size):
            
            match batch['domain'][i]:
                case "finance":
                    assign(domain, 'finance', batch, i, batch_ground_truths, batch_predictions)
                case "music":
                    assign(domain, 'music', batch, i, batch_ground_truths, batch_predictions)
                case "movie": 
                    assign(domain, 'movie', batch, i, batch_ground_truths, batch_predictions)
                case "sports": 
                    assign(domain, 'sports', batch, i, batch_ground_truths, batch_predictions)
                case "open":
                    assign(domain, 'open', batch, i, batch_ground_truths, batch_predictions)
                case default: 
                    print("Wrong Field in domain!")
                          
            match batch['question_type'][i]:
                case "simple":
                    assign(question_type, "simple", batch, i, batch_ground_truths, batch_predictions)                    
                case "simple_w_condition":
                    assign(question_type, "simple_w_condition", batch, i, batch_ground_truths, batch_predictions)
                case "comparison": 
                    assign(question_type, "comparison", batch, i, batch_ground_truths, batch_predictions)
                case "aggregation": 
                    assign(question_type, "aggregation", batch, i, batch_ground_truths, batch_predictions)
                case "set":
                    assign(question_type, "set", batch, i, batch_ground_truths, batch_predictions)
                case "false_premise":
                    assign(question_type, "false_premise", batch, i, batch_ground_truths, batch_predictions)
                case "post-processing":
                    assign(question_type, "post-processing", batch, i, batch_ground_truths, batch_predictions)
                case "multi-hop":
                    assign(question_type, "multi-hop", batch, i, batch_ground_truths, batch_predictions)
                case default: 
                    print("Wrong Field in question_type!")
                          
            match batch['static_or_dynamic'][i]:
                case "static": 
                    assign(static_or_dynamic, "static", batch, i, batch_ground_truths, batch_predictions)
                case "slow-changing": 
                    assign(static_or_dynamic, "slow-changing", batch, i, batch_ground_truths, batch_predictions)
                case "fast-changing": 
                    assign(static_or_dynamic, "fast-changing", batch, i, batch_ground_truths, batch_predictions)
                case "real-time": 
                    assign(static_or_dynamic, "real-time", batch, i, batch_ground_truths, batch_predictions)
                case default: 
                    print("Wrong Field in static_or_dynamic!")
            
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    
    return domain, question_type, static_or_dynamic, queries, ground_truths, predictions 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="data/crag_task_1_dev_v4_release.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2", # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2", # full data
                                 ])
    parser.add_argument("--split", type=int, default=1,
                        help="The split of the dataset to use. This is only relevant for the full data: "
                             "0 for public validation set, 1 for public test set")

    parser.add_argument("--model_name", type=str, default="rag_baseline",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 "modified_rag"
                                 ],
                        )

    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        choices=["meta-llama/Llama-3.2-3B-Instruct",
                                 "google/gemma-2-2b-it",
                                 "gpt-4",
                                 "gpt-4o-mini"
                                 ])
    parser.add_argument("--is_server", action="store_true", default=True,
                        help="Whether we use vLLM deployed on a server or offline inference.")
    parser.add_argument("--vllm_server", type=str, default="http://localhost:8088/v1",
                        help="URL of the vLLM server if is_server is True. The port number may vary.")

    args = parser.parse_args()
    print(args.is_server)

    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[0]
    split = -1
    if dataset == "data":
        split = args.split
        if split == -1:
            raise ValueError("Please provide a valid split value for the full data: "
                             "0 for public validation set, 1 for public test set.")
    dataset_path = os.path.join("..", dataset_path)

    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]
    
    model_name = args.model_name
    if model_name == "vanilla_baseline":
        from vanilla_baseline import InstructModel
        model = InstructModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server)
    elif model_name == "rag_baseline":
        from rag_baseline import RAGModel
        model = RAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server)
    elif model_name == "modified_rag":
        from modified_rag import MyRAGModel
        model = MyRAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server)
    else:
        raise ValueError("Model name not recognized.")

    # make output directory
    output_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
    os.makedirs(output_directory, exist_ok=True)

    # Generate predictions
    domain, question_type, static_or_dynamic, queries, ground_truths, predictions = generate_predictions(dataset_path, model, split)
    data = {
        "domain": domain, 
        "question_type": question_type, 
        "static_or_dynamic": static_or_dynamic, 
        "queries": queries, 
        "ground_truths": ground_truths, 
        "predictions": predictions
    }
    
    # save predictions
    json.dump(data, open(os.path.join(output_directory, "detailed_predictions.json"), "w"), indent=4)
