<!---
# FKGE:Federated Knowledge Graphs Embedding
Code used for paper [Federated Knowledge Graphs Embedding](https://arxiv.org/abs/2105.07615), we use aligned entities to perform entity embedding translation over 11 knowledge graphs.
--->
## To do:
Data release: The datasets we used for experiments have been partially uploaded.
To obtain all the KGs, you can find it in https://drive.google.com/file/d/1oD1Gv2RbpNzO8GWGq7SusbAmYih5r-6Q/view?usp=sharing

You can run the baseline experiments through the following code: 'python Config.py baseline 300 100 1.0 -1', where you can replace baseline with strategy_1 or strategy_2 to conduct the experiments with respect to FKGE. 

And the other parameters denotes epoches, dimension, gan_ratio and pred_id respectively. 

* Paper: https://arxiv.org/abs/2105.07615
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
