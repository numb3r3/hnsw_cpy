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

Three core functions: `add()`, `query()`.

```python
from hnsw_cpy import HnswIndex

# read from file and build index
fp = open('data.bin', 'rb')

index = HnswIndex(bytes_per_vector=4)
index.add(fp.read())

# build a query of 4 bytes
query = bytes([255, 123, 23, 56])

# find and return matched indices
result = index.query(query)  # List[int]
```
