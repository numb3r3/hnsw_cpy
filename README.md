# hnsw_cpy

A scalable HNSW binary vector index in Cython.

## Install

```bash
pip install src/.
```

## Test

To run all unit tests:

```bash
pip install src/.[test]
python -m unittest tests/*.py
```

## Lint

```bash
pylint src/**/*.py
```


## Usage

Two core functions: `index()`,  `query()`.

```python
from hnsw_cpy import HnswIndex
import numpy as np


bytes_num = 8

index = HnswIndex(bytes_num=bytes_num)

# random generate 10000 vectors
data_size = 10000
for i in range(data_size):
    data = np.random.randint(1, 255, data_dim, dtype=np.uint8).tobytes()
    index.index(i, data)

# build a random query
query = np.random.randint(1, 255, data_dim, dtype=np.uint8).tobytes()

# find the top 10 result
result = index.query(query, 10)
```
