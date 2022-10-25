<!---
# FKGE:Federated Knowledge Graphs Embedding
Code used for paper [Federated Knowledge Graphs Embedding](https://arxiv.org/abs/2105.07615), we use aligned entities to perform entity embedding translation over 11 knowledge graphs.
--->
## Differentially Private Federated Knowledge Graphs Embedding:

### Data Release
The datasets we used for experiments have been partially uploaded.
To obtain all the KGs, you can find it in https://drive.google.com/file/d/1oD1Gv2RbpNzO8GWGq7SusbAmYih5r-6Q/view?usp=sharing.
Make sure to put KGs from the Google Drive into ```OpenKE/benchmarks```.

**Update**: The aligned files are updated and already put in the ```trainse_data/aligned``` folder.

### Package Dependencies
* numpy
* tensorflow 1.xx
* tensorflow_probability

### Baseline Embeddings

**You need to run the baseline experiments to obtain the KG embeddings through the following code**: 

```python Config.py baseline 300 100 1.0 -1```

The parameters denotes mode, epoches, dimension, gan_ratio and pred_id respectively. 

Note that if you want to try other embedding algorithms or some files like ```1-1.txt``` is missing, you need to run ```n_n.py``` from ```OpenKE/benchmarks``` for each KG in ```/OpenKE/benchmarks/KG_1```.
You can replace baseline with strategy_1 or strategy_2 to conduct the experiments with respect to FKGE. 

By running baseline embeddings, you will create a ```experiment/``` folder and the embeddings are inside ```experiment/0/``` if you sepcify ```pred_id=-1```.


### Federated Knowledge Graphs Embedding

**After obtaining KG's initital embeddings from running the baseline model (make sure there are embeddings in the ```experiment/0/``` folder), run**: 

```python Config.py strategy_1 300 100 1.0 0```

### DPFKGE
If you want to train FKGE with the *PATE* mechanism, in `Config.py`, replace 

```from FederalTransferLearning.hetro_AGCN_mul_dataset import GAN```

with

```from FederalTransferLearning.hetro_AGCN_mul_dataset_pate import GAN```


### Citation
* Paper: https://arxiv.org/abs/2105.07615

If you use this code in your work, please kindly cite it.

```
@inproceedings{Peng-2021-DPFKGE,
  title={Differentially Private Federated Knowledge Graphs Embedding},
  author={Hao Peng and
          Haoran Li and
          Yangqiu Song and
          Vincent W. Zheng and
          Jianxin Li},
  booktitle={CIKM 2021},
  year={2021},
  url={https://arxiv.org/abs/2105.07615}
}
```
