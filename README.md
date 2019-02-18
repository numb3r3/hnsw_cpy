# BTIndex

[![BK Pipelines Status](https://api.bkdevops.qq.com/process/api/external/pipelines/projects/btindex/p-9e8e4c8ef61242b49b66a83b1b712e11/badge?X-DEVOPS-PROJECT-ID=btindex)](http://api.devops.oa.com/process/api-html/user/builds/projects/btindex/pipelines/p-9e8e4c8ef61242b49b66a83b1b712e11/latestFinished?X-DEVOPS-PROJECT-ID=btindex)

A scalable binary vector index in Cython.

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

Three core functions: `add()`, `find()`, `contains()`.

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