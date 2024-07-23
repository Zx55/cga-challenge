# RH20T-P API

#### Introduction

We provide basic API for RH20T-P dataset, including reading multi-view/multi-frame image/depth data and corresponding annotation (`ee_pos/ee_pos_2d`).

You can further develop on the basis of this API, for example, integrating them into the conversation of LLM to construct an instruction tuning dataset.

#### Download

|Sources|URL|
|:---:|:---:|
|RH20T dataset|[download](https://rh20t.github.io/#download)|
|RH20T-P annotation|[download](https://drive.google.com/file/d/1ssNJikkaEYViz4yr-vIdQjmWoqiLWuwz/view?usp=sharing)|

#### Usage

```python
from data.rh20tp.dataset import (
    LazyRH20TPrimitiveDataset, 
    LazyRH20TActionDataset
)

dataset = LazyRH20TPrimitiveDataset(
    data_root='/path/to/rh20t/dataset',
    anno_path='/path/to/rh20tp/annotation')
```

Here, `data_root` refers to the path of RH20T dataset, e.g., `'data/sources/RH20T'`; `anno_path` refers to the path of RH20T-P annotation file, e.g., `'data/sources/rh20tp_cga_metadata_v1.0.pkl'`.