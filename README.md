## 🚀 Project Setup & Run Guide

Follow these steps to set up and run the project:

### 1. 📥 Clone the Repository

Clone the repository and switch to the correct branch:

```bash
git clone https://github.com/Sidhaarth-Murali/hrl-reasoning.git
cd hrl-reasoning
git checkout bi-level-score
```

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
pip install deepteam
pip install -U transformers
```

### 3. 🔐 Authenticate with Hugging Face

```bash
huggingface-cli login
```

When prompted, enter your Hugging Face token:

```
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
```

### 4. 🧪 Set Up Weights & Biases (wandb)

Set your API key for wandb:

```bash
WANDB_API_KEY=6fdbd76f324713e888458cab04538c67e0ae8505
```

Enter (2) when prompted on run, and copy paste the above API KEY.

### 5. ▶️ Run the Script

Execute the main script:

```bash
./run_parallel.sh
```

---

Make sure your environment has access to required hardware (e.g., GPUs) and internet connectivity for model downloads and logging.
