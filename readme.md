Commands to Run Unlearning & Evaluation

# 1. Run unlearning training on TOFU datasets
C:\Users\ameyg\anaconda3\python.exe trainer.py

# 2. Run benchmark evaluation (Base Model vs Unlearned Model)
C:\Users\ameyg\anaconda3\python.exe TOFU/extra/evaluate_tofu.py

# 3. Quick sample evaluation (e.g. 5 items for fast verification)
C:\Users\ameyg\anaconda3\python.exe TOFU/extra/evaluate_tofu.py --max_examples 5

# 4. Test interactive generation on unlearned model
C:\Users\ameyg\anaconda3\python.exe inference.py --prompt "Who is Jaime Vasquez?"
