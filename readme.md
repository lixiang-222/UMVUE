## UMVUE-DR

This is the official pytorch implementation of "UMVUE-DR: Uniformly Minimum Variance Unbiased Doubly Robust Learning with Reduced Bias and Variance for Debiased Recommendation" paper.
We use three public real-world datasets (**Coat**, **Yahoo! R3** and **KuaiRec**) for experiments. 

## Environment
The code runs well at python 3.8.10. The required packages are as follows:
```bash
pytorch == 2.0.0
numpy == 1.24.2
scipy == 1.10.1
pandas == 2.0.0
```

## Project Structure
- `run.py`: Main execution file with hyperparameter optimization
- `matrix_factorization_ori.py`: Implementation of various matrix factorization models including UMVUE-DR
- `dataset.py`: Data loading and preprocessing utilities
- `utils.py`: Utility functions for evaluation metrics and data processing
- `data/`: Directory containing all datasets (Coat, Yahoo! R3, KuaiRec)
- `saved/`: Directory for storing precomputed embeddings and KNN matrices
- `results/`: Directory for storing experiment results
- `run_experiments.sh`: Script for running experiments with different sampling rates

### Running experiments:
```bash
python run.py --dataset coat --batch_size 128
python run.py --dataset yahoo --batch_size 1024
python run.py --dataset kuai --batch_size 512
```