# BiANE
Codes for our SIGIR 2020 paper [BiANE: Bipartite Attributed Network Embedding](https://dl.acm.org/doi/abs/10.1145/3397271.3401068)

## Dataset
Dataset should be processed as following:

```user_id.tsv: [user_name, '\t', ,user_id]```, user node id; (user_id should start from 0)

```item_id.tsv: [item_name, '\t', ,item_id]```, item node id; (item_id should start from 0)

```adjlist_user_id.tsv: [user_name, '\t', ,user id for adjlist]```, user node ids of the adjacency list file; (adjlist_user_id should start from 0, which is exactly the same as user_id.)

```adjlist_item_id.tsv: [item_name, '\t', ,item_id for adjlist]```, item node ids of the adjacency list file; (It's suggested that the adjlist_item_id should start from the end of adjlist_user_id. For instance, if adjlist_user_id is from 0 to 100, the adjlist_item_id should start from 101.)

```adjlist.txt: [node_itself neighbor_node_0 neighbor_node_1 nerighbor_node_2 neighbor_node_3 ... neighbor_node_k]```, the adjacency list for the graph (training set), each node is represented as its adjlist id;

```train.tsv: [user_id, item_id, label]```, the dataset for training link prediction model (logistic regression model);

```train.csv: [user_id, item_id]```, the dataset for training model;

```valid.tsv: [user_id, item_id, label]```, the validation set;

```test.tsv: [user_id, item_id, label]```, the test set;

```user_attr.pkl: user_attr[user_id][:] ```, a matrix of user attributes;

```item_attr.pkl: item_attr[item_id][:]```, a matrix of item attributes;

```
emb.txt:
        node_number, dimension (skip this line)
        <\s>(invalid token), embedding (skip this line)
        node_adjlist_id, embedding
        ......                   
```
, a matrix of high-order structure features for nodes. Each node is adjlist id. 
This file is the output of [metapath2vec++](https://ericdongyx.github.io/metapath2vec/m2v.html).

## Usage
### Installation
Please refer to [Non-Metric Space Library (NMSLIB)](https://github.com/nmslib/nmslib) for HNSW installation.