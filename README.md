# BTIndex

A scalable binary vector index in Cython.

## Install

```bash
pip install src/. -U
```

## Test

To run all unit tests:

```bash
python -m unittest tests/*.py
```

## Usage

Three core functions: `add`, `find`, `contains`.

```python
from bindex import BIndex

# read from file and build index
fp = open('data.bin', 'rb')
bt = BIndex(bytes_per_vector=4)
bt.add(fp.read())

# build a query of 4 bytes
query = bytes([255, 123, 23, 56])

# find and return matched indices
idx = bt.find(query)  # List[int]
```