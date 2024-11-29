import openai
from loguru import logger

class EvaluationModel:
    def __init__(self, llm_name="gpt-3.5-turbo", openai_api_key=None, max_retries=3):
        self.llm_name = llm_name
        self.max_retries = max_retries

        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')

        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable or pass 'openai_api_key' parameter.")

    def evaluate(self, prompt, max_tokens=10):
        for attempt in range(self.max_retries):
            try:
                # Old `ChatCompletion` call for `openai==0.28`
                response = openai.ChatCompletion.create(
                    model=self.llm_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    n=1,
                    stop=None
                )
                # Extract the reply content
                reply = response["choices"][0]["message"]["content"].strip()
                return reply
            except Exception as e:
                logger.warning(f"Error during OpenAI API call: {e}. Attempt {attempt + 1}/{self.max_retries}")
        logger.error(f"Failed to get response from OpenAI API after {self.max_retries} attempts.")
        return None
