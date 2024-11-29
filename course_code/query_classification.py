import argparse
import json
import os
import time
from loguru import logger
from tqdm.auto import tqdm
import openai
import re

# Refined descriptions for the 8 query types, with one example each
QUERY_TYPE_DESCRIPTIONS = """
Simple question: Questions asking for simple facts, such as the birth date of a person and the authors of a book.
Example: "Who wrote 'To Kill a Mockingbird'?"

Simple question with some condition: Questions asking for simple facts with some given conditions, such as stock price on a certain date and a director's recent movies in a certain genre.
Example: "What was the closing price of Tesla stock on June 1, 2021?"

Set question: Questions that expect a set of entities or objects as the answer. An example is what are the continents in the southern hemisphere?
Example: "List all the Nobel Peace Prize winners from 2000 to 2010."

Comparison question: Questions that may compare two entities, such as who started performing earlier, Adele or Ed Sheeran?
Example: "Which country is larger in area, Canada or the United States?"

Aggregation question: Questions that may need aggregation of retrieval results to answer, for example, how many Oscar awards did Meryl Streep win?
Example: "How many goals did Lionel Messi score in the 2019 season?"

Multi-hop question: Questions that may require chaining multiple pieces of information to compose the answer, such as who acted in Ang Lee's latest movie?
Example: "Who is the current CEO of the company that owns Instagram?"

Post-processing question: Questions that need reasoning or processing of the retrieved information to obtain the answer, for instance, How many days did Thurgood Marshall serve as a Supreme Court justice?
Example: "If I have 250 apples and distribute them equally among 5 baskets, how many apples will be in each basket?"

False Premise question: Questions that have a false proposition or assumption; for example, What's the name of Taylor Swift's rap album before she transitioned to pop? (Taylor Swift didn't release any rap album.)
Example: "When did the Roman Empire land on the Moon?" (The Roman Empire didn't exist during the era of space exploration.)
"""

# Valid query types for validation
VALID_TYPES = [
    "simple",
    "simple_w_condition",
    "set",
    "comparison",
    "aggregation",
    "multi-hop",
    "post-processing",
    "false premise"
]

def sanitize_classification(raw_response):
    """
    Map raw model response to valid classification types.
    """
    response = raw_response.strip().lower()

    # Remove any surrounding quotes or punctuation
    response = response.strip('\'"!.?')

    # Direct mapping with potential synonyms
    mapping = {
        "simple question": "simple",
        "simple questions": "simple",
        "simple": "simple",
        "simple with condition": "simple_w_condition",
        "simple question with some condition": "simple_w_condition",
        "simple questions with some condition": "simple_w_condition",
        "simple with some condition": "simple_w_condition",
        "set question": "set",
        "set questions": "set",
        "set": "set",
        "comparison question": "comparison",
        "comparison questions": "comparison",
        "comparison": "comparison",
        "aggregation question": "aggregation",
        "aggregation questions": "aggregation",
        "aggregation": "aggregation",
        "multi-hop question": "multi-hop",
        "multi-hop questions": "multi-hop",
        "multi-hop": "multi-hop",
        "post-processing question": "post-processing",
        "post-processing questions": "post-processing",
        "post-processing": "post-processing",
        "false premise question": "false premise",
        "false premise questions": "false premise",
        "false premise": "false premise"
    }

    # Attempt exact match
    if response in mapping:
        return mapping[response]

    # Attempt partial match
    for key in mapping:
        if key in response:
            return mapping[key]

    return "unknown"

def fallback_classification(query):
    """
    A simple heuristic-based fallback classification to reduce 'unknown' responses.
    This can be expanded based on common patterns in your data.
    """
    query = query.lower()

    if any(word in query for word in ["who", "what", "when", "where", "which"]):
        if "list" in query or "all" in query:
            return "set"
        elif "how many" in query or "number of" in query:
            return "aggregation"
        elif "compare" in query or "larger" in query or "smaller" in query or "more than" in query:
            return "comparison"
        elif "if" in query or "distribute" in query or "equally" in query or "how much" in query:
            return "post-processing"
        elif "ceo" in query or "current" in query or "executive" in query:
            return "multi-hop"
        else:
            return "simple"
    elif any(word in query for word in ["false", "didn't", "not exist", "impossible", "never"]):
        return "false premise"
    else:
        return "unknown"

def classify_queries(dataset_path, output_path, model, max_retries=5, backoff_factor=2, initial_delay=1):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    with open(dataset_path, "r") as f:
        data = json.load(f)

    question_types = data.get("question_type", {})
    if not question_types:
        logger.error("The 'question_type' field is missing or empty in the dataset. Exiting.")
        return

    queries_with_ground_truths = []
    for query_type, items in question_types.items():
        queries = items.get("queries", [])
        for query in queries:
            queries_with_ground_truths.append((query, query_type))

    logger.info(f"Classifying {len(queries_with_ground_truths)} queries from {dataset_path}.")

    classified_results = []

    # Pre-define few-shot examples
    few_shot_examples = [
        ('"Who wrote \'To Kill a Mockingbird\'?"', 'simple'),
        ('"What was the closing price of Tesla stock on June 1, 2021?"', 'simple_w_condition'),
        ('"List all the Nobel Peace Prize winners from 2000 to 2010."', 'set'),
        ('"Which country is larger in area, Canada or the United States?"', 'comparison'),
        ('"How many goals did Lionel Messi score in the 2019 season?"', 'aggregation'),
        ('"Who is the current CEO of the company that owns Instagram?"', 'multi-hop'),
        ('"If I have 250 apples and distribute them equally among 5 baskets, how many apples will be in each basket?"', 'post-processing'),
        ('"When did the Roman Empire land on the Moon?"', 'false premise')
    ]

    # Pre-format the few-shot examples
    few_shot_formatted = ""
    for i, (query, classification) in enumerate(few_shot_examples, 1):
        few_shot_formatted += f"{i}. Query: {query}\n   Classification: {classification}\n\n"

    for query, ground_truth_type in tqdm(queries_with_ground_truths, desc="Classifying Queries"):
        formatted_prompt = f"""
You are a highly accurate question classifier. Your task is to classify a given query into one of the following categories:

{QUERY_TYPE_DESCRIPTIONS}

Here are some examples to help you understand the classification:

{few_shot_formatted}

Now, classify the following query:

Query: "{query}"

Please provide only the category name as your response.
"""

        response = None
        attempt = 0
        delay = initial_delay
        while attempt < max_retries:
            try:
                response = model.evaluate(
                    prompt=formatted_prompt,
                    max_tokens=10
                )
                logger.debug(f"Raw model response for query '{query}': {response}")
                break
            except openai.error.RateLimitError as e:
                wait_time = delay
                logger.warning(f"Rate limit reached: {e}. Waiting for {wait_time} seconds before retrying. Attempt {attempt + 1}/{max_retries}.")
                time.sleep(wait_time)
                delay *= backoff_factor
                attempt += 1
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}. Marking query as 'unknown'.")
                response = None
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Marking query as 'unknown'.")
                response = None
                break

        if response is None:
            classification = "unknown"
        else:
            classification = sanitize_classification(response)

        if classification not in VALID_TYPES:
            logger.warning(f"Invalid classification '{classification}' for query '{query}'. Attempting fallback classification.")
            # Fallback Mechanism: Simple Heuristic
            classification = fallback_classification(query)
            if classification not in VALID_TYPES:
                classification = "unknown"

        classified_results.append({
            "query": query,
            "classification": classification,
            "ground_truth_type": ground_truth_type
        })

        if classification != ground_truth_type:
            logger.debug(f"Misclassification - Query: '{query}' | Classified as: '{classification}' | Ground Truth: '{ground_truth_type}'")

    # Save results
    with open(output_path, "w") as f:
        json.dump(classified_results, f, indent=4)

    logger.info(f"Classified results saved to {output_path}.")

    # Calculate accuracy
    correct_classifications = sum(
        1 for result in classified_results if result["classification"] == result["ground_truth_type"]
    )
    total_queries = len(classified_results)
    accuracy = correct_classifications / total_queries * 100
    logger.info(f"Classification Accuracy: {accuracy:.2f}%")

    # Debugging: Print sample outputs
    logger.info("Sample results:")
    for result in classified_results[:5]:  # Adjust number of samples to print
        logger.info(f"Query: {result['query']} | Classification: {result['classification']} | Ground Truth: {result['ground_truth_type']}")

class EvaluationModel:
    def __init__(self, llm_name, openai_api_key):
        openai.api_key = openai_api_key
        self.llm_name = llm_name

    def evaluate(self, prompt, max_tokens):
        response = openai.ChatCompletion.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0  # Set temperature to 0 for deterministic output
        )
        return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    DEFAULT_DATASET_PATH = os.path.join("..", "output", "data", "rag_baseline", "Llama-3.2-3B-Instruct", "detailed_predictions.json")
    DEFAULT_OUTPUT_PATH = "/home/jupyter/cs245-project-crag-master/output/classified_queries.json"
    DEFAULT_LLM_NAME = "gpt-3.5-turbo"  # Updated to GPT-3.5-turbo

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to the input dataset.")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save the classified queries.")
    parser.add_argument("--llm_name", type=str, default=DEFAULT_LLM_NAME, help="Name or path to the language model.")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key.")
    args = parser.parse_args()

    model = EvaluationModel(llm_name=args.llm_name, openai_api_key=args.openai_api_key)
    classify_queries(dataset_path=args.dataset_path, output_path=args.output_path, model=model)
