# ESR-Bench Kaggle Templates

This folder contains Kaggle-ready task notebook templates for the Social Cognition submission.

Files:
- `esr_bench_task1_notebook.py`
- `esr_bench_task2_notebook.py`

## How to use on Kaggle

1. Create a new task notebook at `https://www.kaggle.com/benchmarks/tasks/new`.
2. Paste the contents of `esr_bench_task1_notebook.py` into one notebook.
3. Paste the contents of `esr_bench_task2_notebook.py` into a second notebook.
4. Upload or attach the gold CSVs so the notebook can access:
   - `task1_gold.csv`
   - `task2_gold.csv`
5. Run the notebook top to bottom. Each notebook now:
   - loads the gold CSV
   - defines a single-row subtask with `store_task=False`
   - evaluates the full dataset with `.evaluate(...)`
   - runs the main aggregate benchmark task once with `kbench.llm`
6. In the final notebook cell, run:
   - `%choose esr_task1_benchmark` for Task 1
   - `%choose esr_task2_benchmark` for Task 2
7. Save each task.
8. Create a Kaggle Benchmark and add both saved tasks.

## Expected data paths

The notebook templates try these paths in order:
- `/kaggle/input/esr-bench-gold/task1_gold.csv`
- `/kaggle/input/datasets/annajsims/esr-bench-gold/task1_gold.csv`
- `/kaggle/input/esr-bench-gold/task2_gold.csv`
- `/kaggle/input/datasets/annajsims/esr-bench-gold/task2_gold.csv`
- `benchmark_data/task1_gold.csv`
- `benchmark_data/task2_gold.csv`

## Why two notebooks

Kaggle Benchmarks is cleaner to manage when each underlying task has its own task notebook and result history.
