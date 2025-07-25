# OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs

This is an implementation of the OpenScholar paper (arXiv:2411.14199v1), which presents a specialized retrieval-augmented language model for synthesizing scientific literature.

## Overview

OpenScholar consists of:

- **OpenScholar-DataStore (OSDS)**: A datastore of 45 million open-access papers
- **Specialized Retrievers**: Bi-encoder retriever and cross-encoder reranker trained on scientific data
- **Self-Feedback Inference**: Iterative refinement process for improving response quality
- **OpenScholar-8B**: An 8B parameter model trained on synthetic data

## Features

- 📚 Retrieves from 45M scientific papers
- 🔄 Self-feedback loop for iterative improvement
- 📊 Citation-backed responses with high accuracy
- 🎯 Outperforms GPT-4o and PaperQA2 on ScholarQABench
- 🚀 Efficient 8B model option

## Installation

```bash
# Clone the repository
git clone https://github.com/raghav-rama/openscholar.git
cd openscholar

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Download Pre-trained Models (Optional)

```bash
# Download retriever model
wget https://huggingface.co/OpenScholar/retriever/resolve/main/pytorch_model.bin -P models/retriever/

# Download generator model
wget https://huggingface.co/OpenScholar/generator-8b/resolve/main/pytorch_model.bin -P models/generator/
```

### 2. Run the Demo

```bash
# Start the Streamlit demo
streamlit run app.py
```

### 3. Use the Pipeline Programmatically

```python
from openscholar import (
    OpenScholarDataStore,
    OpenScholarRetriever,
    OpenScholarReranker,
    OpenScholarGenerator,
    OpenScholarPipeline
)

# Initialize components
datastore = OpenScholarDataStore("data/openscholar_datastore")
datastore.load()

retriever = OpenScholarRetriever()
reranker = OpenScholarReranker()
generator = OpenScholarGenerator()

# Create pipeline
pipeline = OpenScholarPipeline(
    datastore=datastore,
    retriever=retriever,
    reranker=reranker,
    generator=generator
)

# Generate response
response = pipeline.generate("What are recent advances in neural architecture search?")
print(response.response)
```

## Training Your Own Models

### 1. Prepare the DataStore

```bash
# Create training configuration
python scripts/train_openscholar.py --create_config config/training_config.json

# Edit the configuration file as needed
# Then run the training pipeline
python scripts/train_openscholar.py --config config/training_config.json
```

### 2. Train Individual Components

```bash
# Train retriever
python scripts/train_retriever.py \
    --train_data data/retriever_train.json \
    --model_name facebook/contriever \
    --output_dir models/my_retriever

# Train generator
python scripts/train_generator.py \
    --train_data data/generator_train.json \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir models/my_generator
```

## Evaluation

### ScholarQABench Evaluation

```python
from openscholar.evaluation import ScholarQABench, OpenScholarEvaluator

# Load benchmark
benchmark = ScholarQABench("data/scholarqa_bench.json")

# Generate predictions
predictions = []
for query in benchmark.get_queries():
    response = pipeline.generate(query)
    predictions.append({
        "query_id": query.query_id,
        "response": response.response,
        "citations": response.citations
    })

# Evaluate
evaluator = OpenScholarEvaluator()
results = evaluator.evaluate_dataset(
    predictions,
    benchmark,
    save_path="results/evaluation.json"
)

print(f"Citation Accuracy: {results['citation_accuracy_mean']:.3f}")
print(f"Factual Correctness: {results['factual_correctness_mean']:.3f}")
print(f"Coverage: {results['coverage_mean']:.3f}")
```

## Project Structure

```
openscholar/
├── src/openscholar/
│   ├── __init__.py
│   ├── datastore.py         # DataStore implementation
│   ├── retriever.py         # Retriever and Reranker
│   ├── generator.py         # Generator with self-feedback
│   ├── pipeline.py          # Main pipeline
│   ├── data_generation.py   # Synthetic data generation
│   └── evaluation.py        # Evaluation metrics
├── scripts/
│   ├── train_retriever.py   # Retriever training
│   ├── train_generator.py   # Generator training
│   └── train_openscholar.py # Full training pipeline
├── app.py                   # Streamlit demo
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Configuration

See `config/demo_config.json` for an example configuration. Key parameters:

- `datastore.path`: Path to the datastore
- `retriever.model_name`: Retriever model to use
- `generator.model_name`: Generator model to use
- `pipeline.final_k`: Number of passages for generation

## Citation

If you use OpenScholar in your research, please cite:

```bibtex
@article{asai2024openscholar,
  title={OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs},
  author={Asai, Akari and He, Jacqueline and Shao, Rulin and others},
  journal={arXiv preprint arXiv:2411.14199},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the OpenScholar paper by Asai et al. (2024).
