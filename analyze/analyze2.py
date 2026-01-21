# Analyzing our optimized profiling results.
# The profile2.txt file was generated using cProfile in train.py with the following command:
#       uv run python -m cProfile -o profile2.txt ./src/ml_ops_project/train.py

# Afterwads, we executed this analyze2.py to be able to analyze the results.
import pstats
p = pstats.Stats('analyze/profile2.txt')
p.sort_stats('cumulative').print_stats(10)
