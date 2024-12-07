import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLSectionSplitter
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from langchain.schema import Document
from openai import OpenAI
from datetime import datetime
from dateutil import parser
from pytz import timezone, UTC
from tqdm import tqdm

# ### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 64

NUM_CONTEXT_SENTENCES2 = 32
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1500
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 35000

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

# ### CONFIG PARAMETERS END---

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source, page_time):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Define the text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP
        )
        
        documents = [Document(page_content=text)]
        # Split the text sections into smaller chunks
        splits = text_splitter.split_documents(documents)

        # Convert the split result into a list of chunks
        chunks = [("Page Last Modified : " + str(page_time) + "\n" + split.page_content) for split in splits]

        # Return the interaction ID and the generated chunks
        return interaction_id, chunks


    def get_time_difference(self,q1, q2):
        # Define the formats of the input timestamps
        format_q1 = "%m/%d/%Y, %H:%M:%S %Z"  # Correct format for q1
        format_q2 = "%a, %d %b %Y %H:%M:%S %Z"  # Format for q2

        # Define the time zones
        pt_zone = timezone("US/Pacific")
        gmt_zone = UTC  # GMT is equivalent to UTC

        # Parse the timestamps with their respective time zones
        try:
            dt_q1 = parser.parse(q1).replace(tzinfo=pt_zone)
            dt_q2 = parser.parse(q2).replace(tzinfo=gmt_zone)
        except ValueError as e:
            return f"Error parsing date: {e}"

        # Calculate the difference
        time_difference = dt_q1 - dt_q2

        # Return the difference in various formats
        return int(time_difference.total_seconds() / 3600)

    def extract_chunks(self, batch_interaction_ids, batch_search_results,query_mask,query_times):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = []

        for idx, search_results in enumerate(batch_search_results):
            if query_mask[idx]=='real-time' or query_mask[idx]=='fast-changing':  # Apply filtering based on query_mask
                query_time = query_times[idx]
                print("Query Time:", query_time)
                nearest_page = None
                nearest_time_difference = 24
                if query_mask[idx]=='fast-changing':
                    nearest_time_difference = 168
                

                for html_text in search_results:
                    page_time = html_text.get("page_last_modified")
                    print("Page Time:", page_time)

                    if not page_time:  # If page_time is empty or None, include it in refs
                        print("Page Time is empty, adding to refs")
                        ray_response_refs.append(
                            self._extract_chunks.remote(
                                self,
                                interaction_id=batch_interaction_ids[idx],
                                html_source=html_text["page_result"],
                                page_time=page_time,  # Can pass None or an empty string
                            )
                        )
                        continue  # Skip further processing for this page

                    try:
                        time_difference = self.get_time_difference(query_time, page_time)
                        print("Time Difference:", time_difference)
                        if 0 <= time_difference < nearest_time_difference:
                            nearest_time_difference = time_difference
                            nearest_page = html_text
                    except ValueError:
                        continue  # Skip if time parsing fails

                if nearest_page:
                    ray_response_refs.append(
                        self._extract_chunks.remote(
                            self,
                            interaction_id=batch_interaction_ids[idx],
                            html_source=nearest_page["page_result"],
                            page_time=nearest_page["page_last_modified"],
                        )
                    )
            else:  # No filtering, process all pages
                for html_text in search_results:
                    ray_response_refs.append(
                        self._extract_chunks.remote(
                            self,
                            interaction_id=batch_interaction_ids[idx],
                            html_source=html_text["page_result"],
                            page_time=html_text["page_last_modified"],
                        )
                    )

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        # chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class MyRAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "sk-proj-utwYZXwGAgD2kpygKAQIT3BlbkFJB9U9lluAod4BZAWFCVG1"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        # self.sentence_model = BGEM3FlagModel('BAAI/bge-m3',  
        #     use_fp16=True,
        #     device=torch.device(
        #         "cuda" if torch.cuda.is_available() else "cpu"
        #     )
        # )
        self.sentence_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1',
                                                  device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ))
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True,device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ))

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        # print(sentences[:5])
        # print(len(sentences))
        # Ensure chunks is a list of strings
        if not isinstance(sentences, list):
            raise ValueError(f"Expected chunks to be a list of strings, got {type(sentences)}")
        if not all(isinstance(chunk, str) for chunk in sentences):
            raise ValueError("All elements in chunks must be strings.")
        # embeddings = self.sentence_model.encode(
        #     sentences,
        #     batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE
        # )['dense_vecs']
        embeddings = self.sentence_model.encode(
            sentences,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size
    
    def classification(self, query):
        # Define the prompt with clear definitions and examples
        prompt = (
            "Classify the following query into one of the temporal dynamics categories:\n"
            "- Real-time: Requires information that changes moment to moment (e.g., current stock prices).\n"
            "- Fast-changing: Information changes event to event, such as daily or weekly updates (e.g., sports match outcomes).\n"
            "- Slow-changing: Information evolves gradually over months or years (e.g., product features).\n"
            "- Stable: Information remains largely unchanged over long periods (e.g., historical facts).\n\n"
            "Examples:\n"
            "1. Query: \"Can you provide me with the most recent stock price of CURO today?\"\n"
            "   Classification: Real-time\n"
            "2. Query: \"What was the outcome of Sheffield Utd's most recent match in the Premier League?\"\n"
            "   Classification: Fast-changing\n"
            "3. Query: \"What are the major features of the iPhone 15?\"\n"
            "   Classification: Slow-changing\n"
            "4. Query: \"What is the chemical formula for water?\"\n"
            "   Classification: Stable\n\n"
            f"Now, classify the following query:\n"
            f"Query: \"{query}\"\n"
            "Classification: "
        )
        print(query)
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                top_p=0.25,
                temperature=0,
                max_tokens=5,
            )

            answer = response.choices[0].message.content
            if 'real-time' in answer.lower() :
                return 'real-time'
            elif 'fast-changing' in answer.lower() :
                return 'fast-changing'
            else:
                return 'stable'
        else:
            raise ValueError("Server mode is not enabled.")
        
        
    def extract_answer(self,input_str):
        """
        Extracts the text after 'Answer:' from the given input string.

        Args:
            input_str (str): The input string containing the "Answer:" field.

        Returns:
            str: The extracted answer text or an empty string if 'Answer:' is not found.
        """
        answer_keyword = "Answer:"
        try:
            # Find the position of "Answer:"
            start_index = input_str.index(answer_keyword) + len(answer_keyword)
            # Extract and strip any leading/trailing whitespace
            # print("--------------- Output ---------------------")
            # print(input_str)
            # print(input_str[start_index:].strip())
            return input_str[start_index:].strip()

        except ValueError:
            # Return empty string if "Answer:" is not found
            return "I don't know"
    
    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        
        query_mask = [self.classification(query) for query in queries]
        print(query_mask)
        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results,query_mask,query_times
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]
            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = np.array([chunk for i, chunk in enumerate(chunks) if relevant_chunks_mask[i]])
            relevant_chunks_embeddings = np.array([embedding for i, embedding in enumerate(chunk_embeddings) if relevant_chunks_mask[i]])


#             relevant_pairs = [[query,i] for i in relevant_chunks]

#             cosine_scores = self.sentence_model.compute_score(relevant_pairs, 
#                           max_passage_length=512, # a smaller max length leads to a lower latency
#                           weights_for_different_modes=[0.6, 0.4,0])

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings @ query_embedding.T)
            # print(cosine_scores)
            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            
            scores = []
            for chunk in retrieval_results:
                score = self.reranker.compute_score([query, chunk], normalize=True)
                scores.append(score)
                
            scores = -1 * np.array(scores)
            scores = scores.squeeze()
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            
            reranked_results = retrieval_results[
                (scores).argsort()[:NUM_CONTEXT_SENTENCES2]
            ]
            
            batch_retrieval_results.append(reranked_results)
            
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.25,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=200,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [self.extract_answer(response.choices[0].message.content)]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=200,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(self.extract_answer(response.outputs[0].text))

        return answers
    
    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
    #     system_prompt = '''You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible.
    # ## Strictly follow these Instructions:
    # 1. If the references do not contain the necessary information to answer the question, respond with only 'i don't' know'.
    # 2. If the Question seems to be a false premise , output "invalid question". like Questions that have a false proposition or assumption; for example, What's the name of Taylor Swift's rap album before she transitioned to pop? (Taylor Swift didn't release any rap album.)
    # 3. There is no need to explain the reasoning behind your answers , give exact and to the point in the answer . No need to repeat the question or give reasoning 
    # 4. Read the references carefully and find the answer only from provided references & output the to the point answer verbatim from the references. Do not use any other information'''
        system_prompt = '''You are provided with a question and various references. Your task is to answer the question based on the references, categorizing it into the appropriate type, and reasoning step-by-step to reach the final answer.

        ## Instructions (Read it Very Carefully):
        1. Categorize the Question: First, determine the type of question from the following categories:
           - Simple Question : Questions asking for simple facts
           - Simple Question with Conditions : Questions asking for simple facts with some given conditions, such as stock price on a certain date and a director's recent movies in a certain genre.
           - Set Question : Questions that expect a set of entities or objects as the answer
           - Comparison Question : Questions that may compare two entities, such as who started performing earlier, Adele or Ed Sheeran?
           - Aggregation Question : Questions that may need aggregation of retrieval results to answer, for example, how many Oscar awards did Meryl Streep win?
           - Multi-hop Question : Questions that may require chaining multiple pieces of information to compose the answer, such as who acted in Ang Lee's latest movie?
           - Post-processing Question : Questions that need reasoning or processing of the retrieved information to obtain the answer
           - False Premise Question :  Questions that have a false proposition or assumption; for example, What's the name of Taylor Swift's rap album before she transitioned to pop? (Taylor Swift didn't release any rap album.)

        2. Address Recent or Fast-Changing Information:
           - For questions asking about recent or fast-changing information, use the query time provided (time of asking query) and the "Page Last Modified" for each reference chunk.
           - Only use the reference chunks with modification times closest to the query time.
        
        3. Multi-Step Reasoning:
           - Some questions might not have a direct answer in the references and will require combining multiple pieces of information through logical reasoning.
           - Always connect intermediate steps explicitly to arrive at the final answer.
        
        4. Provide Chain-of-Thought Reasoning: Use logical reasoning to connect the information in the references to the final answer.
        
        5. Evaluate References: Use only the provided references to answer the question. If the references do not provide sufficient information or no information to deduce the answer, respond with:
           Answer: I don't know

        6. Validate Premises: If the question is based on a false premise, respond with:
           Answer: Invalid question
           
        7. Answer Format: Always provide the output in the following structure:
        Question Type: <type>
        Reasoning: <step-by-step reasoning>
        Answer: <final answer>

        ### Examples:

        1. Simple Question
           Question: Who wrote *To Kill a Mockingbird*?
           References: ["Harper Lee wrote *To Kill a Mockingbird*."]

           Output:
           Question Type: Simple Question
           Reasoning: The references state that Harper Lee is the author.
           Answer: Harper Lee

        2. Simple Question with Conditions
           Question: What was the closing price of Tesla stock on June 1, 2021?
           References: ["On June 1, 2021, Tesla's closing stock price was $605.13."]

           Output:
           Question Type: Simple Question with Conditions
           Reasoning: The references show the stock price on June 1, 2021, was $605.13.
           Answer: $605.13

        3. Set Question
           Question: List all the Nobel Peace Prize winners from 2000 to 2010.
           References: ["Winners from 2000 to 2010 are X, Y, Z."]

           Output:
           Question Type: Set Question
           Reasoning: The references list winners for each year in that range: X, Y, Z.
           Answer: X, Y, Z

        4. Comparison Question
           Question: Which country is larger in area, Canada or the United States?
           References: ["Canada is 9.98 million km²; the U.S. is 9.83 million km²."]

           Output:
           Question Type: Comparison Question
           Reasoning: The references show Canada is 9.98 million km² and the U.S. is 9.83 million km².
           Answer: Canada

        5. Aggregation Question
           Question: How many goals did Lionel Messi score in the 2019 season?
           References: ["Messi scored 25 in the league, 10 in the Champions League, and 5 in other tournaments."]

           Output:
           Question Type: Aggregation Question
           Reasoning: The references indicate Messi scored 25 in the league, 10 in the Champions League, and 5 in other tournaments, totaling 40 goals.
           Answer: 40 goals

        6. Multi-hop Question
           Question: Who is the current CEO of the company that owns Instagram?
           References: ["Instagram is owned by Meta.", "Meta's CEO is Mark Zuckerberg."]

           Output:
           Question Type: Multi-hop Question
           Reasoning: Instagram is owned by Meta, whose CEO is Mark Zuckerberg.
           Answer: Mark Zuckerberg

        7. Post-processing Question
           Question: If I have 250 apples and distribute them equally among 5 baskets, how many apples will be in each basket?
           References: ["Divide 250 by 5 to get 50."]

           Output:
           Question Type: Post-processing Question
           Reasoning: Divide 250 by 5. The result is 50.
           Answer: 50

        8. False Premise Question
           Question: When did the Roman Empire land on the Moon?
           References: ["The Roman Empire did not exist during the era of space exploration."]

           Output:
           Question Type: False Premise Question
           Reasoning: The Roman Empire did not exist during the space exploration era.
           Answer: Invalid question
        
        9. Fast-Changing Information Question
           Question: What was the closing price of Tesla stock yesterday?
           Current Time: 02/28/2024, 08:30:00 PT
           References:
           - ["Tesla stock closed at $210.35 on February 27, 2024. (Page Last Modified: 02/28/2024, 06:00:00 PT)"]
           - ["Tesla stock closed at $206.35 on February 26, 2024. (Page Last Modified: 02/27/2024, 05:00:00 PT)"]

           Output:
           Question Type: Simple Question with Conditions
           Reasoning: The query asks for Tesla's closing price on February 27, 2024. The reference modified on 02/28/2024 at 06:00:00 PT provides the latest data and states the closing price was $210.35.
           Answer: $210.35
           
        ### Important Notes:
        - If references are insufficient to answer, respond with:
          Answer: I don't know
        - Always adhere strictly to this format.
        '''

        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            
            user_message = ""
            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"# Reference  {_snippet_idx} \n - {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------ END OF REFERENCES \n\n"
            user_message 
            user_message += f"Strictly use only the references listed above and no other information, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n Question Type: "

            if self.is_server:
                # print(user_message)
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
