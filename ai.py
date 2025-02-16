import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, List, Dict
import os
import json
import arxiv
import wikipediaapi
from scholarly import scholarly
import logging
import asyncio
import aiohttp
from dataclasses import dataclass
from ratelimit import limits, sleep_and_retry
import hashlib
import pronouncing
import warnings

# Suppress non-critical warnings and set logging to only show errors.
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class config:
    model_name: str = "gpt2-large"
    block_size: int = 128
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    max_memory: str = "6GB"
    seed: int = 42
    max_sources_per_query: int = 50
    cache_dir: str = "./data_cache"
    api_keys: dict = None
    # Note: arXiv will only be queried if the user prompt starts with "search "

class data:
    def __init__(self, config: config, tokenizer: GPT2Tokenizer):
        self.config = config
        self.wiki_api = wikipediaapi.Wikipedia(
            user_agent="MyChatbot/1.0 (contact@example.com)",
            language="en"
        )
        self.session = aiohttp.ClientSession()
        os.makedirs(config.cache_dir, exist_ok=True)
        self.tokenizer = tokenizer

    @sleep_and_retry
    @limits(calls=30, period=60)
    async def fetch_arxiv_papers(self, query: str) -> List[Dict]:
        """Fetch relevant papers from arXiv."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.config.max_sources_per_query,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for result in client.results(search):
            papers.append({
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'url': result.pdf_url,
                'published': str(result.published)
            })
        return papers

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def fetch_wikipedia_content(self, query: str) -> List[Dict]:
        page = self.wiki_api.page(query)
        if page.exists() and "may refer to" not in page.summary.lower():
            return [{
                'title': page.title,
                'content': page.text,
                'url': page.fullurl,
                'summary': page.summary
            }]
        return []

    @sleep_and_retry
    @limits(calls=50, period=60)
    async def fetch_google_scholar(self, query: str) -> List[Dict]:
        search_query = scholarly.search_pubs(query)
        papers = []
        try:
            for i in range(min(10, self.config.max_sources_per_query)):
                paper = next(search_query)
                papers.append({
                    'title': paper.get('title'),
                    'abstract': paper.get('abstract'),
                    'authors': paper.get('author'),
                    'url': paper.get('url'),
                    'year': paper.get('year')
                })
        except StopIteration:
            pass
        return papers

    async def close(self):
        await self.session.close()

class EnhancedDataset:
    """Dataset containing data from multiple sources."""
    def __init__(self, data_manager: data, query: str):
        self.data_manager = data_manager
        self.query = query
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        self.cache_file = os.path.join(data_manager.config.cache_dir, f"cache_{query_hash}.json")
        
    async def gather_data(self) -> List[Dict]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        
        tasks = []
        # Only fetch arXiv papers if the query starts with "search "
        if self.query.lower().startswith("search "):
            # Remove the "search " prefix for the arXiv query.
            arxiv_query = self.query[len("search "):].strip()
            tasks.append(self.data_manager.fetch_arxiv_papers(arxiv_query))
        # Always include Wikipedia and Google Scholar
        tasks.append(self.data_manager.fetch_wikipedia_content(self.query))
        tasks.append(self.data_manager.fetch_google_scholar(self.query))
        
        results = await asyncio.gather(*tasks)
        combined_data = []
        for source_data in results:
            if source_data:
                combined_data.extend(source_data)
        with open(self.cache_file, 'w') as f:
            json.dump(combined_data, f, default=str)
        return combined_data

    def __iter__(self):
        loop = asyncio.new_event_loop()
        data = loop.run_until_complete(self.gather_data())
        loop.close()
        for item in data:
            if item is not None:
                snippet = item.get('abstract') or item.get('summary') or item.get('content') or ''
                text = f"{item.get('title', '')} {snippet}"
                tokens = self.data_manager.tokenizer.encode(text)
                yield torch.tensor(tokens, dtype=torch.long)

class chat:
    """Chatbot with real-time data integration and conversational ability."""
    
    def __init__(self, model_dir: Optional[str] = None, config: Optional[config] = None):
        self.config = config or config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_identifier = self.config.model_name if model_dir is None else model_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_identifier).to(self.device)
        self.data_manager = data(self.config, self.tokenizer)
        self.context = []

    def get_pronunciation_response(self, user_query: str) -> str:
        vocabulary = ["hello", "name", "pronunciation", "chatbot", "science", "research", "data", "artificial", "intelligence"]
        response_lines = []
        for word in vocabulary:
            phones = pronouncing.phones_for_word(word)
            if phones:
                response_lines.append(f"{word}: {phones[0]}")
            else:
                response_lines.append(f"{word}: [No pronunciation found]")
        return "Here are some words and their pronunciations:\n" + "\n".join(response_lines)

    async def research_topic(self, query: str) -> List[Dict]:
        # If the query does not start with "search " and is too short, skip research.
        if not query.lower().startswith("search ") and len(query.split()) < 20:
            return []
        dataset = EnhancedDataset(self.data_manager, query)
        data = await dataset.gather_data()
        if not data:
            return []
        return data

    async def generate_informed_response(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        conversation_context = "\n".join(self.context) if self.context else ""
        user_query = prompt
        
        if "pronounce my name" in user_query.lower():
            custom_response = "I don't know. It sounds like 'I don't know'.\n"
            custom_response += self.get_pronunciation_response(user_query)
            self.context.append(f"You: {prompt}")
            self.context.append(f"Chatbot: {custom_response}")
            return custom_response

        research_data = await self.research_topic(user_query)
        
        # Initialize enhanced_prompt with a default response if no research data is found
        enhanced_prompt = (
            f"{conversation_context}\n"
            "Answer the question concisely and naturally, without any extra explanations.\n"
            f"Query: {prompt}\n"
            "Response:"
        )
        
        if research_data:
            context_snippets = "\n".join([
                f"Title: {item.get('title', '')}\nSummary: {(item.get('abstract') or item.get('summary') or item.get('content') or '')[:200]}..."
                for item in research_data[:5] if item is not None
            ])
            
            enhanced_prompt = (
                f"{conversation_context}\n"
                f"Research Context: {context_snippets}\n"
                "Answer the question concisely and naturally, without any extra explanations.\n"
                f"Query: {prompt}\n"
                "Response:"
            )
            
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5, 
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                top_k=10,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        if outputs is None or outputs.shape[0] == 0:
            final_response = "I'm not sure."
        else:
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parts = decoded.split("Response:")
            if len(parts) > 1:
                final_response = parts[-1].strip()
            else:
                final_response = decoded.strip()
    
        self.context.append(f"You: {prompt}")
        self.context.append(f"Chatbot: {final_response}")
        return final_response

async def run_chatbot(config: config = config()):
    try:
        chatbot = chat(config=config)
        print("Chatbot: Hello! I can help you with information from various academic and scientific sources.")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Chatbot: Goodbye!")
                    break
                prompt = f"user: {user_input}"
                response = await chatbot.generate_informed_response(prompt)
                print(f"Chatbot: {response}")
            except KeyboardInterrupt:
                print("\nChatbot: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print("Chatbot: I encountered an error. Please try again.")
        await chatbot.data_manager.close()
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        print("Failed to start chatbot. Please check the logs.")

if __name__ == "__main__":
    config = config()
    asyncio.run(run_chatbot(config))
