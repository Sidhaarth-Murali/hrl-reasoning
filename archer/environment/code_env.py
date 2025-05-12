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
INITIAL_STR = ""
DEFAULT_DATASET_PATH = "dataset/mbpp_filtered.csv"


class CodeDataset:
    """Loader for code programming problems"""
    def __init__(self, data_path: str = DEFAULT_DATASET_PATH, test_size: int = 4, seed: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.seed = seed
        self.train_problems = []
        self.train_solutions = []
        self.train_test_cases = []
        self.test_problems = []
        self.test_solutions = []
        self.test_test_cases = []
        self.load_data()
        
    def load_data(self):
        """Load problems and solutions from CSV dataset file"""    
        if not os.path.exists(self.data_path):

            raise FileNotFoundError(f"Code dataset path not found: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)

            rename_map = {}
            if 'text' in df.columns:
                rename_map['text'] = 'problem'
            if 'code' in df.columns:
                rename_map['code'] = 'reference_solution'
            if 'test_list' in df.columns:
                rename_map['test_list'] = 'test_cases'
                
            if rename_map:
                df = df.rename(columns=rename_map)
                
            required_cols = ['problem', 'reference_solution']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file must have columns: {required_cols}")
            
            # If test_cases column is missing, create an empty list for each row
            if 'test_cases' not in df.columns:
                df['test_cases'] = [[] for _ in range(len(df))]
            
            # Process test_cases if they're stored as strings
            if df['test_cases'].dtype == 'object':
                # Try parsing as JSON list first
                try:
                    df['test_cases'] = df['test_cases'].apply(json.loads)
                except:
                    # Try parsing each item (in case the whole column isn't a valid JSON)
                    try:
                        df['test_cases'] = df['test_cases'].apply(
                            lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
                        )
                    except:
                        # If not JSON, try splitting by pipe character
                        df['test_cases'] = df['test_cases'].apply(
                            lambda x: x.split('|') if isinstance(x, str) else []
                        )
            
            # Add setup code to problems if present
            if 'test_setup_code' in df.columns:
                for i, row in df.iterrows():
                    if pd.notna(row['test_setup_code']) and row['test_setup_code']:
                        setup = row['test_setup_code']
                        # Add setup code to problem description
                        df.at[i, 'problem'] = (
                            df.at[i, 'problem'] + 
                            "\n\nSetup code:\n```python\n" + 
                            setup + 
                            "\n```"
                        )
            
            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
            # Split into train and test
            test_df = df.iloc[:self.test_size]
            train_df = df.iloc[self.test_size:]
            
            # Load train data
            self.train_problems = train_df['problem'].tolist()
            self.train_solutions = train_df['reference_solution'].tolist()
            self.train_test_cases = train_df['test_cases'].tolist()
            
            # Load test data
            self.test_problems = test_df['problem'].tolist()
            self.test_solutions = test_df['reference_solution'].tolist()
            self.test_test_cases = test_df['test_cases'].tolist()
            
            print(f"Loaded {len(self.train_problems)} training problems and {len(self.test_problems)} test problems")
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise


class CodeEnv():
    def __init__(
        self,
        max_tokens: int=1024,
        data_path: str=DEFAULT_DATASET_PATH,
        test_mode: bool=False,
        language: str="python"
    ):
        self.max_tokens = max_tokens
        self.test_mode = test_mode
        self.language = language
        self.curr_problem = None
        self.curr_solution = None
        self.curr_test_cases = None
        self.history = ''
        self.done = True
        self.token_count = 0
        self.eos_str = "\n"
                
        # Load dataset
        self._load_dataset(data_path)
    
    def _load_dataset(self, data_path: str):
        """Load the Code dataset"""
        try:
            dataset = CodeDataset(data_path)
            if self.test_mode:
                self.problems = dataset.test_problems
                self.solutions = dataset.test_solutions
                self.test_cases = dataset.test_test_cases
            else:
                self.problems = dataset.train_problems
                self.solutions = dataset.train_solutions
                self.test_cases = dataset.train_test_cases
            
            if not self.problems:
                raise ValueError("No problems loaded from dataset")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def is_correct(self, generated_code):
        """Evaluate the generated code by running test cases."""
        try:
            # Extract code from the model's output
            code_pattern = r"```(?:python|javascript)?\s*([\s\S]*?)```"
            code_match = re.search(code_pattern, generated_code)
            
            if code_match:
                code = code_match.group(1).strip()
            else:
                # If no code block is found, use the entire answer
                code = generated_code.strip()
            
            # Run test cases
            test_results = self.evaluate_code(code, self.curr_test_cases)
            
            # Calculate reward based on test case results
            # Modified: Rewards should be between [-1, 1] instead of [0, 1]
            if test_results["total"] == 0:
                # If no test cases, fall back to exact match
                normalized_code = self._normalize_code(code)
                normalized_solution = self._normalize_code(self.curr_solution)
                exact_match = normalized_code == normalized_solution
                reward = 1.0 if exact_match else -1.0
            else:
                # Scale from -1.0 (all failed) to 1.0 (all passed)
                correct_ratio = test_results["correct"] / test_results["total"]
                reward = 2.0 * correct_ratio - 1.0  # Transforms [0,1] to [-1,1]
                
            # Store results for potential feedback
            self.last_evaluation_results = {
                "code": code,
                "reference_solution": self.curr_solution,
                "test_results": test_results,
                "reward": reward
            }
            
            # For step() compatibility, return if all tests passed
            all_tests_passed = test_results["correct"] == test_results["total"]
            return all_tests_passed, reward
        except Exception as e:
            self.last_evaluation_results = {
                "code": generated_code,
                "error": str(e),
                "reward": -1.0  # Modified: use -1.0 instead of 0.0 for errors
            }
            return False, -1.0
    
    def evaluate_code(self, code_str, test_cases):
        """Run test cases against the code and return results."""
        results = {
            "correct": 0,
            "total": len(test_cases),
            "details": []
        }
        
        if len(test_cases) == 0:
            return results
        
        # Create a clean local scope for execution
        local_scope = {}
        
        # Redirect stdout to capture prints
        import io
        import sys
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # First execute the code to define the functions
            exec(code_str, globals(), local_scope)
            
            # Then run each test case
            for i, test_case in enumerate(test_cases):
                try:
                    # Execute the test case (should be an assert statement)
                    exec(test_case, globals(), local_scope)
                    # If no assertion error, test passed
                    results["correct"] += 1
                    results["details"].append({
                        "test_case": test_case,
                        "passed": True
                    })
                except AssertionError:
                    results["details"].append({
                        "test_case": test_case,
                        "passed": False,
                        "error": "Assertion failed"
                    })
                except Exception as e:
                    results["details"].append({
                        "test_case": test_case,
                        "passed": False,
                        "error": str(e)
                    })
        except Exception as e:
            # Error in code execution (syntax error, etc.)
            results["details"].append({
                "error": f"Code execution failed: {str(e)}"
            })
        finally:
            # Restore stdout
            sys.stdout = original_stdout
        
        return results
    
    def _normalize_code(self, code):
        """Normalize code for comparison by removing whitespace and comments."""
        # Remove all whitespace
        code = re.sub(r'\s+', '', code)
        # Remove all comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        return code
    
    def get_feedback(self):
        """Return feedback based on the last evaluation results."""
        if not hasattr(self, 'last_evaluation_results'):
            return "No evaluation results available."
        
        results = self.last_evaluation_results
        
        if "error" in results:
            return f"Error occurred"
        
        test_results = results["test_results"]
        feedback = f"Passed {test_results['correct']}/{test_results['total']} test cases."
        
        return feedback
    
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
        all_passed, reward = self.is_correct(self.history)
                
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
            self.curr_solution = self.solutions[idx]
            self.curr_test_cases = self.test_cases[idx]
        else:
            idx = random.randint(0, len(self.problems)-1)
            self.curr_problem = self.problems[idx]
            self.curr_solution = self.solutions[idx]
            self.curr_test_cases = self.test_cases[idx]
            
        self.history = INITIAL_STR + self.curr_problem + '\n'
        self.done = False
        return self.history

    def copy(self):
        env = CodeEnv(max_tokens=self.max_tokens, language=self.language)
        env.problems = self.problems
        env.solutions = self.solutions
        env.test_cases = self.test_cases
        return env


class LLMBatchedCodeEnv():
    def __init__(
        self,
        env_load_path: str = None,
        cache_dir: str = '~/.cache',
        device = None,
        max_tokens: int=1024,
        bsize: int=128,  
        data_path: str=DEFAULT_DATASET_PATH,
        language: str="python",
        correction_model_path: str = None,  
        use_smart_corrections: bool = True, 
        train_guidance_model: bool = True,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    ):
        # Initialize base environments
        base_env = CodeEnv(max_tokens=max_tokens, data_path=data_path, language=language)
        self.env_list = [base_env.copy() for _ in range(bsize)]
        self.bsize = bsize
        
        # Load model and tokenizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Smart correction settings
        self.use_smart_corrections = use_smart_corrections
        self.train_guidance_model = train_guidance_model
        
        # Initialize correction model
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
                    self.correction_model = None
            else:
                # Use the same model for both tasks (to save memory)
                self.correction_model = self.model
                print("Using the same model for both solution generation and correction guidance")
        else:
            self.correction_model = None
        
        # Load custom weights if provided
        if env_load_path:
            try:
                self.model.load_state_dict(torch.load(env_load_path, map_location=self.device)['model_state_dict'])
                print(f"Loaded model weights from {env_load_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
    
    def generate_correction_instruction(self, problem, solution):
        """Generate a custom correction instruction for a given problem and solution"""
        from archer.prompts.code import generate_smart_correction_prompt

        if self.use_smart_corrections and self.correction_model:   
            # If the model has a generate_custom_guidance method (our RL trainer), use it
            if hasattr(self.correction_model, 'generate_custom_guidance'):
                try:
                    guidance_texts, _ = self.correction_model.generate_custom_guidance([problem], [solution])
                    return guidance_texts[0]
                except Exception as e:
                    print(f"Error using RL-trained guidance: {e}")
                    # Fall back to standard smart correction
 
            return generate_smart_correction_prompt(
                problem, 
                solution, 
                correction_model=self.correction_model,
                tokenizer=self.tokenizer,
                device=self.correction_model.device,
                language=self.env_list[0].language
            )
        else:
            # Fall back to static template
            from archer.prompts.code import format_code_self_correction_prompt
            return format_code_self_correction_prompt(problem + solution, language=self.env_list[0].language)

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
        
    def get_feedback(self):
        """Get feedback from all environments"""
        return [env.get_feedback() if hasattr(env, 'get_feedback') else "" for env in self.env_list]
    
    def get_rewards(self):
        """Get the latest rewards from all environments"""
        rewards = []
        for env in self.env_list:
            if hasattr(env, 'last_evaluation_results') and 'reward' in env.last_evaluation_results:
                rewards.append(env.last_evaluation_results['reward'])
            else:
                rewards.append(0.0)
        return rewards 