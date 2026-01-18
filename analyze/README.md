## M12: Profiling and Optimization

We used cProfile to analyze our training code and identify performance bottlenecks.

### Step 1: Baseline Profiling

First, we profiled the original code:
```bash
uv run python -m cProfile -o analyze/profile1.txt ./src/ml_ops_project/train.py
```

The profiling results showed that PIL Image decoding and conversions were the main bottleneck, taking approximately 107 seconds of the total 113.088 seconds.

### Step 2: Optimization

After analyzing the profiling results, we implemented the tensor pre-loading optimization from the course exercise on cProfile.

### Step 3: Re-profiling

After optimizing the code, we re-profiled:
```bash
uv run python -m cProfile -o analyze/profile2.txt ./src/ml_ops_project/train.py
```

Afterwards, we executed analyze2.py to analyze the results.

### Results

The code went from taking 113.088 seconds to 21.741 seconds, achieving a **5.2x speedup**.

All related files can be found in the `analyze/` folder.
