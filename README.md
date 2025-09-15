# Automated Short Answer Grading (ASAG) with SBERT & Meta-Learning

Automated Short Answer Grading (ASAG) system for Bahasa Indonesia essay questions using Sentence-BERT embeddings and meta-learning.

## Results
- SMAPE: **6.34%** (vs 13.09% supervised baseline)
- Demonstrated strong generalization across open- and close-ended question types.

## Features
- Embeddings with IndoSBERT and Multilingual SBERT
- Reptile meta-learning for few-shot grading scenarios
- Benchmarked loss functions: MSE vs SMAPE
- Evaluation pipeline for essay-type answers

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Train model
```bash
python src/train.py   --train data/train.jsonl   --val data/val.jsonl   --model indobenchmark/indobert-base-p1   --meta reptile   --loss smape   --epochs 50
```

### Evaluate
```bash
python src/evaluate.py   --test data/test.jsonl   --model checkpoints/best_model.pt   --metrics mse smape
```

### Inference
```bash
python src/infer.py   --model checkpoints/best_model.pt   --input "Jelaskan perbedaan supervised dan unsupervised learning."   --target_score 10
```

## Limitations
- Dataset not included due to copyright. Provide your own annotated essays.
- Performance sensitive to annotation consistency and domain.
- Requires GPU for efficient training.

## Roadmap
- Cross-lingual grading with Multilingual SBERT
- Transformer-based meta-learners (MAML, ProtoNet)
- Web demo for teachers/students
- Dockerfile and CI/CD integration

## Citation
```
@software{ASAG_SBERT_MetaLearning,
  title   = {Automated Short Answer Grading with SBERT & Meta-Learning},
  author  = {Rian Aprilyanto Siburian},
  year    = {2024},
  url     = {https://github.com/aprilrian/ASAG}
}
```

## License
MIT License. See `LICENSE` for details.

## Acknowledgements
- [Sentence-BERT](https://www.sbert.net/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)
