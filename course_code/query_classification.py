import argparse
import json
import os
import asyncio
from loguru import logger
import sys
from tqdm.asyncio import tqdm
import openai
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rapidfuzz import process, fuzz

# Configure loguru to log both to stderr and a file
logger.remove()
logger.add(sys.stderr, level="DEBUG")  # Console logging
logger.add("classification_debug.log", rotation="10 MB", level="DEBUG")  # File logging

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Log NLTK data paths for debugging
logger.debug(f"NLTK data paths: {nltk.data.path}")

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
    Map raw model response to valid classification types using exact matching and fuzzy matching.
    """
    # Extract the classification category from the response using regex
    # This handles cases where the model might include prefixes like 'Classification: '
    match = re.search(r'\b(simple_w_condition|simple|set|comparison|aggregation|multi-hop|post-processing|false premise)\b', raw_response.lower())
    if match:
        classification = match.group(1)
        if classification in VALID_TYPES:
            return classification

    # If no exact match found, proceed with fuzzy matching
    response = raw_response.strip().lower()
    response = response.strip('\'"!.?')

    # Check if the response is too long (unlikely to be a single category)
    if len(response.split()) > 5:  # Adjusted for longer categories
        return "unknown"

    # Mapping of possible responses to valid types
    mapping = {
        "simple question with some condition": "simple_w_condition",
        "simple question": "simple",
        "simple": "simple",
        "set question": "set",
        "set": "set",
        "comparison question": "comparison",
        "comparison": "comparison",
        "aggregation question": "aggregation",
        "aggregation": "aggregation",
        "multi-hop question": "multi-hop",
        "multi-hop": "multi-hop",
        "post-processing question": "post-processing",
        "post-processing": "post-processing",
        "false premise question": "false premise",
        "false premise": "false premise"
    }

    # Sort choices by length descending to prioritize longer matches
    choices = sorted(mapping.keys(), key=lambda x: len(x), reverse=True)

    # Use fuzzy matching to find the best match
    result = process.extractOne(response, choices, scorer=fuzz.partial_ratio)

    logger.debug(f"sanitize_classification received result: {result}")

    if result is not None:
        if isinstance(result, tuple):
            if len(result) >= 2:
                match, score = result[:2]
            else:
                logger.error(f"Unexpected result format: {result}")
                return "unknown"
        elif hasattr(result, 'string') and hasattr(result, 'score'):
            # For RapidFuzz versions that return ExtractResult
            match, score = result.string, result.score
        else:
            logger.error(f"Unexpected result type: {result}")
            return "unknown"

        # Ensure the matched string is at the start of the response to avoid partial matches
        if score >= 80 and response.startswith(match):
            return mapping.get(match, "unknown")

    return "unknown"

def fallback_classification(query):
    """
    Enhanced heuristic-based fallback classification to reduce 'unknown' responses.
    """
    query = query.lower()
    tokens = word_tokenize(query)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Define keyword sets for each category
    keywords = {
        "simple": {"who", "what", "when", "where", "which"},
        "simple_w_condition": {"price", "revenue", "condition", "date", "status", "value"},
        "set": {"list", "name", "all", "entities", "set", "group"},
        "comparison": {"compare", "larger", "smaller", "more", "less", "than"},
        "aggregation": {"how many", "number of", "total", "sum", "aggregate", "count"},
        "multi-hop": {"ceo", "director", "executive", "current", "owner", "manager"},
        "post-processing": {"if", "distribute", "equally", "divide", "spread", "calculate"},
        "false premise": {"didn't", "not exist", "impossible", "never", "invalid", "fictional"}
    }

    # Check for false premise first as it's more definitive
    for keyword in keywords["false premise"]:
        if keyword in query:
            return "false premise"

    # Check for other categories based on keywords
    for category, kw_set in keywords.items():
        if category == "false premise":
            continue  # Already checked
        if any(kw in query for kw in kw_set):
            return category

    # Default to 'unknown' if no keywords matched
    return "unknown"

async def classify_queries_async(dataset_path, output_path, model, max_retries=5, backoff_factor=2, initial_delay=1, concurrency=5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset file not found at path: {dataset_path}. Exiting.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from the dataset file: {e}. Exiting.")
        return

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
    semaphore = asyncio.Semaphore(concurrency)

    # Initialize a cache dictionary
    query_cache = {}

    # Pre-define expanded few-shot examples with more diversity
    few_shot_examples = [
        ('"Who wrote \'To Kill a Mockingbird\'?"', 'simple'),
        ('"What was the closing price of Tesla stock on June 1, 2021?"', 'simple_w_condition'),
        ('"List all the Nobel Peace Prize winners from 2000 to 2010."', 'set'),
        ('"Which country is larger in area, Canada or the United States?"', 'comparison'),
        ('"How many goals did Lionel Messi score in the 2019 season?"', 'aggregation'),
        ('"Who is the current CEO of the company that owns Instagram?"', 'multi-hop'),
        ('"If I have 250 apples and distribute them equally among 5 baskets, how many apples will be in each basket?"', 'post-processing'),
        ('"When did the Roman Empire land on the Moon?"', 'false premise'),
        # Additional examples for each category
        ('"What is the capital of France?"', 'simple'),
        ('"What was Apple\'s revenue in Q4 2022?"', 'simple_w_condition'),
        ('"Name all the countries in the European Union."', 'set'),
        ('"Is Python more popular than Java?"', 'comparison'),
        ('"How many Olympic medals has Usain Bolt won?"', 'aggregation'),
        ('"Who directed the latest Marvel movie?"', 'multi-hop'),
        ('"If you buy 3 shirts at $20 each, how much will you spend?"', 'post-processing'),
        ('"Did Albert Einstein win a Grammy Award?"', 'false premise')
    ]

    # Pre-format the few-shot examples
    few_shot_formatted = ""
    for i, (query, classification) in enumerate(few_shot_examples, 1):
        few_shot_formatted += f"{i}. Query: {query}\n   Classification: {classification}\n\n"

    async def classify_single_query(query, ground_truth_type):
        nonlocal classified_results
        if query in query_cache:
            classification = query_cache[query]
        else:
            async with semaphore:
                formatted_prompt = f"""
You are a highly accurate question classifier. Your task is to classify a given query into one of the following categories:

{QUERY_TYPE_DESCRIPTIONS}

Here are some examples:

{few_shot_formatted}

Query: "{query}"
Classification (provide only the category name):
"""

                response = None
                attempt = 0
                delay = initial_delay
                while attempt < max_retries:
                    try:
                        response = await model.evaluate(
                            prompt=formatted_prompt,
                            max_tokens=10
                        )
                        logger.debug(f"Raw model response for query '{query}': {response}")
                        break
                    except openai.error.RateLimitError as e:
                        wait_time = delay
                        logger.warning(f"Rate limit reached: {e}. Waiting for {wait_time} seconds before retrying. Attempt {attempt + 1}/{max_retries}.")
                        await asyncio.sleep(wait_time)
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
                    # First fallback: heuristic-based
                    classification = fallback_classification(query)
                    if classification not in VALID_TYPES:
                        classification = "unknown"

                # Cache the result
                query_cache[query] = classification

        classified_results.append({
            "query": query,
            "classification": classification,
            "ground_truth_type": ground_truth_type
        })

        if classification != ground_truth_type:
            logger.debug(f"Misclassification - Query: '{query}' | Classified as: '{classification}' | Ground Truth: '{ground_truth_type}'")
            # Log misclassification for analysis
            try:
                with open("misclassifications.log", "a") as mis_log:
                    mis_log.write(f"Query: '{query}' | Classified as: '{classification}' | Ground Truth: '{ground_truth_type}'\n")
            except Exception as e:
                logger.error(f"Failed to write to misclassifications.log: {e}")

    # Create a list of tasks
    tasks = [classify_single_query(query, gt) for query, gt in queries_with_ground_truths]

    # Run tasks concurrently with a progress bar
    try:
        await tqdm.gather(*tasks)
    except Exception as e:
        logger.error(f"An error occurred during classification: {e}")

    # Save results
    try:
        with open(output_path, "w") as f:
            json.dump(classified_results, f, indent=4)
        logger.info(f"Classified results saved to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save classified results: {e}")

    # Calculate accuracy
    try:
        correct_classifications = sum(
            1 for result in classified_results if result["classification"] == result["ground_truth_type"]
        )
        total_queries = len(classified_results)
        accuracy = correct_classifications / total_queries * 100
        logger.info(f"Classification Accuracy: {accuracy:.2f}%")
    except ZeroDivisionError:
        logger.error("No queries were classified. Accuracy cannot be calculated.")

    # Debugging: Print sample outputs
    logger.info("Sample results:")
    for result in classified_results[:5]:  # Adjust number of samples to print
        logger.info(f"Query: {result['query']} | Classification: {result['classification']} | Ground Truth: {result['ground_truth_type']}")

class AsyncEvaluationModel:
    def __init__(self, llm_name, openai_api_key):
        openai.api_key = openai_api_key
        self.llm_name = llm_name

    async def evaluate(self, prompt, max_tokens):
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.llm_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0  # Set temperature to 0 for deterministic output
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e

if __name__ == "__main__":
    try:
        DEFAULT_DATASET_PATH = os.path.join("..", "output", "data", "rag_baseline", "Llama-3.2-3B-Instruct", "detailed_predictions.json")
        DEFAULT_OUTPUT_PATH = "/home/jupyter/cs245-project-crag-master/output/classified_queries.json"
        DEFAULT_LLM_NAME = "gpt-3.5-turbo"  # Updated to GPT-3.5-turbo

        parser = argparse.ArgumentParser(description="Classify queries into predefined categories using OpenAI's GPT model.")
        parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to the input dataset.")
        parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save the classified queries.")
        parser.add_argument("--llm_name", type=str, default=DEFAULT_LLM_NAME, help="Name or path to the language model.")
        parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key.")
        args = parser.parse_args()

        model = AsyncEvaluationModel(llm_name=args.llm_name, openai_api_key=args.openai_api_key)
        asyncio.run(classify_queries_async(dataset_path=args.dataset_path, output_path=args.output_path, model=model))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
