# Analyzing our original profiling results.
# The profile1.txt file was generated using cProfile in train.py with the following command:
#       uv run python -m cProfile -o profile1.txt ./src/ml_ops_project/train.py

# Afterwads, we executed this analyze1.py to be able to analyze the results.
import pstats

p = pstats.Stats("analyze/profile1.txt")
p.sort_stats("cumulative").print_stats(10)
