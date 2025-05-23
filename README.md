# EYP-25: Enhance Your Prompt

A comprehensive framework for automatic query enhancement using Small Language Models (SLMs) to improve information retrieval and response quality.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

EYP-25 (Enhance Your Prompt) is a research project that addresses the challenge of poorly formulated user queries in information retrieval systems. The framework automatically enhances vague, incomplete, or ambiguous queries using fine-tuned Small Language Models (SLMs), making them more specific and contextually rich without requiring additional user effort.

### Problem Statement
- Users often provide incomplete or unclear queries
- Traditional systems struggle with vague inputs
- Manual query refinement requires extra user effort
- Poor queries lead to suboptimal search results

### Solution
Our framework provides:
- Automatic query enhancement using SLMs
- Multi-level enhancement strategies (Levels 1-4)
- No additional user interaction required
- Significant improvement in response quality

## ✨ Key Features

- **Automatic Enhancement**: Transforms vague queries into well-structured, specific requests
- **Multi-Level Approach**: Four enhancement levels with increasing sophistication
- **Small Language Models**: Efficient models (GPT-2, T5, Phi-3, Llama-3) that can run on consumer hardware
- **Domain Agnostic**: Works across various domains and query types
- **Evaluation Framework**: Comprehensive metrics for measuring enhancement quality

## 🏗️ Architecture

The EYP-25 framework consists of several key components:

1. **Query Enhancement Pipeline**
   - Input: Original user query
   - Processing: SLM-based enhancement
   - Output: Enhanced query with improved clarity and context

2. **Enhancement Levels**
   - **Level 1**: Basic query improvement
   - **Level 2**: Structural enhancements
   - **Level 3**: Context addition
   - **Level 4**: Comprehensive reformulation

3. **Model Architecture**
   - Decoder-only models: GPT-2, Phi-3, Llama-3
   - Encoder-decoder models: T5

## 📊 Dataset

The project uses the LMSYS+NaturalQuestions dataset, combining:
- LMSYS-Chat-1M dataset (real user queries)
- Natural Questions dataset
- Split into train/validation/test sets

Dataset structure:
```
artifacts/datasets/LMSYS+NQ/
├── train.tsv
├── val.tsv
└── test.tsv
```

Each file contains original queries and their enhanced versions at different levels.

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- Conda package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/EYP-25.git
cd EYP-25
```

2. **Create conda environment**
```bash
conda create --name slm python=3.9 -y
conda activate slm
```

3. **Install dependencies**

For decoder-only models (GPT-2, Phi-3, Llama-3):
```bash
cd code/Finetuning-SLMs
pip install -r requirements.txt
```

For encoder-decoder models (T5):
```bash
cd code/Finetuning-SLMs
pip install -r requirements_T5.txt
```

4. **Setup Redis (for T5 models only)**
```bash
sudo su
chmod +x redis_install.sh
./redis_install.sh
redis-server redis.conf
```

## 💻 Usage

### Fine-tuning Models

1. **Navigate to the specific model directory**
```bash
cd code/Finetuning-SLMs/GPT2_medium  # or T5_large, Phi-3-mini-4k-instruct, etc.
```

2. **Fine-tune the model**
```bash
chmod +x finetune.sh
./finetune.sh
```

For T5 models with multiple GPUs:
```bash
./finetune.sh 8  # for 8 GPUs
```

### Generating Enhanced Queries

1. **For decoder-only models**
```bash
chmod +x generate.sh
./generate.sh
```

2. **For T5 models**

Upload data to Redis first:
```bash
chmod +x automate_upload.sh
./automate_upload.sh ../data
```

Then generate:
```bash
./generate.sh ./checkpoint_path ../data/test.tsv
```

### Using Pre-trained Models

The enhanced queries are available in:
```
artifacts/model_inferences/LMSYS+NQ/{model_name}/level{1-4}.jsonl
```

## 📁 Project Structure

```
EYP-25/
├── README.md
├── CIKM_2025___EYP.pdf          # Research paper
├── code/
│   ├── Readme.md
│   └── Finetuning-SLMs/
│       ├── GPT2_medium/
│       ├── T5_large/
│       ├── Phi-3-mini-4k-instruct/
│       └── Llama-3-8B-instruct/
├── prompts/                      # Enhancement prompts
│   ├── Level 1 Query Enhancement Prompt.txt
│   ├── Level 2 Query Enhancement Prompt.txt
│   ├── Level 3 Query Enhancement Prompt.txt
│   └── Level 4 Query Enhancement Prompt.txt
├── artifacts/
│   ├── datasets/                 # Training/evaluation data
│   ├── model_inferences/         # Model outputs
│   └── agent-inferences/         # Evaluation results
└── models.txt                    # Model configurations
```

## 🤖 Models

### Supported Models
1. **GPT-2 Medium** (355M parameters)
2. **T5 Large** (770M parameters)
3. **Phi-3 Mini 4K** (3.8B parameters)
4. **Llama-3 8B Instruct** (8B parameters)

### Model Selection Criteria
- Computational efficiency
- Enhancement quality
- Resource requirements
- Domain adaptability

## 📈 Evaluation Metrics

The framework uses several metrics to evaluate enhancement quality:

1. **LRQI (Language Response Quality Index)**
   - Measures overall response quality improvement

2. **UI (User Intent Preservation)**
   - Ensures enhanced queries maintain original user intent

3. **EQDQ (Enhanced Query Draft Quality)**
   - Evaluates the quality of enhanced queries

4. **AUE (Additional User Effort)**
   - Measures if enhancement reduces user effort

## 📊 Results

Key findings from our experiments:
- **40% improvement** in response quality with Level 4 enhancements
- **Minimal latency** added to query processing
- **High user intent preservation** (>90% accuracy)
- **Significant reduction** in follow-up queries needed

## 📄 License

**NOTICE: This repository is proprietary and confidential.**
This project and all associated code, documentation, and artifacts are the exclusive property of the authors and are protected under copyright law. 

### Usage Restrictions:
- **No public use, reproduction, or distribution** is permitted without explicit written permission
- **No commercial use** allowed
- **No derivative works** may be created
- **Academic use** requires prior approval and proper attribution
- **Viewing only** - code may be reviewed for research purposes but not executed or modified

Commercial use is strictly prohibited and may result in legal action.

## 🙏 Acknowledgments

- Thanks to the LMSYS and Natural Questions dataset creators
- Microsoft for computational resources
- All contributors and reviewers

**Note**: This is an active research project. Features and documentation may be updated frequently.