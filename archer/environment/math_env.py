import random
import json
import os
import csv
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import concurrent.futures
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval


logging.getLogger().setLevel(logging.CRITICAL)
INITIAL_STR = ""
DEFAULT_DATASET_PATH = "dataset/MATH.csv"
import os
api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

class MathDataset:
    """Loader for MATH dataset problems"""
    def __init__(self, data_path: str = DEFAULT_DATASET_PATH, test_size: int = 500, seed: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.seed = seed
        self.train_problems = []
        self.train_answers = []
        self.train_explanations = []
        self.test_problems = []
        self.test_answers = []
        self.test_explanations = []
        self.load_data()
        
    def load_data(self):
        """Load problems and answers from CSV dataset file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"MATH dataset path not found: {self.data_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(self.data_path)[:5000]
            required_cols = ['question', 'correct_answer', 'answer']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file must have columns: {required_cols}")
            
            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
            # Split into train and test
            test_df = df.iloc[:self.test_size]
            train_df = df.iloc[self.test_size:]
            
            # Load train data
            self.train_problems = train_df['question'].tolist()
            self.train_answers = train_df['answer'].apply(str).tolist()
            self.train_explanations = train_df['correct_answer'].tolist()
            
            # Load test data
            self.test_problems = test_df['question'].tolist()
            self.test_answers = test_df['answer'].apply(str).tolist()
            self.test_explanations = test_df['correct_answer'].tolist()
            
            # Comment out to prevent multiple prints during environment initialization
            # print(f"Loaded {len(self.train_problems)} training problems and {len(self.test_problems)} test problems")
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

class LLMMathEnv():
    def __init__(
        self,
        max_tokens: int=512,
        data_path: str=DEFAULT_DATASET_PATH,
        test_mode: bool=False
    ):
        self.max_tokens = max_tokens
        self.test_mode = test_mode
        self.curr_problem = None
        self.curr_answer = None
        self.history = ''
        self.done = True
        self.token_count = 0
        self.eos_str = "\n"
        
        # Initialize GEval metric for answer verification
        self.correctness_metric = GEval(
            model="o3-mini",
            name="Math Final Answer Correctness",
            evaluation_steps=[
                "Read the actual output and the expected output carefully.",
                "Extract the final numerical answer or boxed result from both.",
                "Compare the two final answers: they must match exactly or be mathematically equivalent.",
                "If the final answer in the actual output is missing, unclear, or incorrect, assign a low score.",
                "Ignore differences in intermediate steps unless they impact the final answer.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=1.0,
            strict_mode=True,
            verbose_mode=False,
        )
        
        # Load dataset
        self._load_dataset(data_path)
    
    def _load_dataset(self, data_path: str):
        """Load the MATH dataset"""
        try:
            dataset = MathDataset(data_path)
            if self.test_mode:
                self.problems = dataset.test_problems
                self.answers = dataset.test_answers
                self.explanations = dataset.test_explanations
            else:
                self.problems = dataset.train_problems
                self.answers = dataset.train_answers
                self.explanations = dataset.train_explanations
            
            if not self.problems:
                raise ValueError("No problems loaded from dataset")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def is_correct(self, generated_answer):
        """Check if the generated answer contains the correct answer number,
        by extracting the number from the last 50 characters of the model's output."""
        
        # 1) Look only at the tail of the answer
        tail = generated_answer
        
        # 2) Extract all numbers in that tail
        nums_in_tail = re.findall(r'-?\d+\.?\d*', tail)
        if not nums_in_tail:
            return False
        
        # 3) Assume the last number in that slice is the model's final answer
        answer_in_generated = nums_in_tail[-1].strip()

        boxed_match = re.search(r'\\boxed{([^}]+)}', self.curr_answer)
        if boxed_match:
            candidate = boxed_match.group(1)
        else:
            candidate = self.curr_answer
        
        # 5) Pull out the first number from that candidate
        target_match = re.search(r'-?\d+\.?\d*', candidate)
        if not target_match:
            return False
        target_answer = target_match.group(0).strip()
        return answer_in_generated == target_answer

    
    def _step(self, action_text):
        """Process a complete multi-token action from the agent."""
        if self.done:
            return None
        new_history = self.history + action_text

        tokenizer = self.get_tokenizer()
        new_tokens_count = len(tokenizer.encode(new_history))
        initial_tokens_count = len(tokenizer.encode(self.history))
        tokens_added = new_tokens_count - initial_tokens_count
        self.token_count += tokens_added
        self.history = new_history

        done = True
        if self.is_correct(self.history):
            reward = 1.0
        else:
            reward = 0.0
        self.done = done
        return self.history, reward, self.done

    def get_tokenizer(self):
        """Get access to the tokenizer"""
        if not hasattr(self, 'tokenizer'):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        return self.tokenizer

    def reset(self, idx: Optional[int]=None):
        self.token_count = 0
        if idx is not None:
            self.curr_problem = self.problems[idx]
            self.curr_answer = self.answers[idx]
        else:
            idx = random.randint(0, len(self.problems)-1)
            self.curr_problem = self.problems[idx]
            self.curr_answer = self.answers[idx]
            
        self.history = INITIAL_STR + self.curr_problem + '\nSolution: '
        self.done = False
        return self.history

    def copy(self):
        env = LLMMathEnv(max_tokens=self.max_tokens)
        env.problems = self.problems
        env.answers = self.answers
        env.explanations = self.explanations
        return env

class LLMBatchedMathEnv():
    def __init__(
        self,
        env_load_path: str = None,
        cache_dir: str = '~/.cache',
        device = None,
        max_tokens: int=512,
        bsize: int=64,  
        data_path: str=DEFAULT_DATASET_PATH,
        correction_model_path: str = None,  
        use_smart_corrections: bool = True, 
    ):
        # Initialize base environments
        base_env = LLMMathEnv(max_tokens=max_tokens, data_path=data_path)
        self.env_list = [base_env.copy() for _ in range(bsize)]
        self.bsize = bsize
        
        # Load model and tokenizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir=cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir=cache_dir)
        
        # Smart correction settings
        self.use_smart_corrections = use_smart_corrections
        self.correction_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir=cache_dir).to(self.device)
        
        # If smart corrections are enabled, load the correction model
        if use_smart_corrections:
            if correction_model_path:
                try:
                    self.correction_model = AutoModelForCausalLM.from_pretrained(
                        correction_model_path, cache_dir=cache_dir
                    ).to(self.device)
                        
                    print(f"Loaded correction model from {correction_model_path}")
                except Exception as e:
                    print(f"Error loading correction model: {e}")
                    print("Falling back to static correction templates")
                    self.use_smart_corrections = False
            else:
                # Use the same model for both tasks (to save memory)
                self.correction_model = self.model
                print("Using the same model for both solution generation and correction guidance")
        
        # Load custom weights if provided
        if env_load_path:
            try:
                self.model.load_state_dict(torch.load(env_load_path, map_location=self.device)['model_state_dict'])
                print(f"Loaded model weights from {env_load_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        
    def generate_tokens(self, states):
        """Generate next token for each state"""
        encoder_ids = self.tokenizer(states, padding=True, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids=encoder_ids['input_ids'], 
            attention_mask=encoder_ids['attention_mask'],
            max_new_tokens=1,  # Generate one token at a time
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        return self.tokenizer.batch_decode(outputs.sequences[:, -1:], skip_special_tokens=True)
        
    def generate_correction_instruction(self, problem, solution):
        """Generate a custom correction instruction for a given problem and solution"""
        from archer.prompts.math import generate_smart_correction_prompt

        if self.use_smart_corrections and self.correction_model:   
 
            return generate_smart_correction_prompt(
                problem, 
                solution, 
                correction_model=self.correction_model,
                tokenizer=self.tokenizer,
                device=self.correction_model.device
            )
        else:
            # Fall back to static template
            from archer.prompts.math import format_math_self_correction_prompt
            return format_math_self_correction_prompt(problem + solution)

    def reset(self, idx: Optional[int] = None):
        """Reset all environments, optionally to a specific problem index"""
        return [env.reset(idx) for env in self.env_list]
    
    def step(self, action_texts):
        """Take a step in all environments using the provided action texts"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env._step, action_text) for env, action_text in zip(self.env_list, action_texts)]
            results = [job.result() for job in jobs]
        return results
        
    def get_current_histories(self):
        """Return the current history of each environment.
        This ensures that agents receive the correct prompts including self-correction instructions.
        """
        return [env.history for env in self.env_list]