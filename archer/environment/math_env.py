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

logging.getLogger().setLevel(logging.CRITICAL)

INITIAL_STR = "Solve the following math problem step-by-step by thinking deeply and concise. When you find the final answer, express it in the format \\boxed{your answer}.\n\nProblem:"
DEFAULT_DATASET_PATH = "dataset/MATH.csv"

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
            
            print(f"Loaded {len(self.train_problems)} training problems and {len(self.test_problems)} test problems")
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

class LLMMathEnv():
    def __init__(
        self,
        max_tokens: int=324,
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

        gen_boxed = re.search(r'\\boxed{([^}]+)}', generated_answer)
        gen_answer = gen_boxed.group(1) if gen_boxed else generated_answer
        
        gen_answer = re.sub(r'\s+', '', gen_answer.lower())
        target_answer = re.sub(r'\s+', '', str(self.curr_answer).lower())
        if gen_answer == target_answer:
            return True

        try:
            from sympy import sympify
            gen_val = float(sympify(gen_answer))
            target_val = float(sympify(target_answer))
            return abs(gen_val - target_val) < 1e-6
        except:
            return target_answer in gen_answer
    
    def _step(self, action_text):
        """Process a complete multi-token action from the agent."""
        if self.done:
            return None
        
        # Append the new action text to the conversation history.
        new_history = self.history + action_text
        
        # Update token count accurately.
        tokenizer = self.get_tokenizer()
        new_tokens_count = len(tokenizer.encode(new_history))
        initial_tokens_count = len(tokenizer.encode(self.history))
        tokens_added = new_tokens_count - initial_tokens_count
        self.token_count += tokens_added
        self.history = new_history

        # If exactly two EOS tokens have been produced, mark the output as complete.
        if self.history.count(self.eos_str) == 2:
            done = True
            # Evaluate the answer: reward 1 if correct, otherwise -1.
            if self.is_correct(self.history):
                reward = 1.0
            else:
                reward = -1.0
        else:
            # Otherwise, check if we've hit the token limit.
            done = (self.token_count >= self.max_tokens)
            if self.is_correct(self.history):
                reward = 1.0
            else:
                reward = -1.0 if done else 0
        
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
            # Randomly select a problem
            idx = random.randint(0, len(self.problems)-1)
            self.curr_problem = self.problems[idx]
            self.curr_answer = self.answers[idx]
            
        # Include "Solution: " in the initial history
        # Better formatting
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
        max_tokens: int=324,
        bsize: int=4,
        data_path: str=DEFAULT_DATASET_PATH,
    ):
        # Initialize base environments
        base_env = LLMMathEnv(max_tokens=max_tokens, data_path=data_path)
        self.env_list = [base_env.copy() for _ in range(bsize)]
        self.bsize = bsize
        
        # Load model and tokenizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir).to(self.device)
        
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

    def reset(self, idx: Optional[int] = None):
        """Reset all environments, optionally to a specific problem index"""
        return [env.reset(idx) for env in self.env_list]
    
    def step(self, action_texts):
        """Take a step in all environments using the provided action texts"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env._step, action_text) for env, action_text in zip(self.env_list, action_texts)]
            results = [job.result() for job in jobs]
        return results