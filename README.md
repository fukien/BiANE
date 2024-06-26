# BiANE
Codes for our SIGIR 2020 paper [BiANE: Bipartite Attributed Network Embedding](https://dl.acm.org/doi/abs/10.1145/3397271.3401068)

## Dataset
Dataset should be processed as following:

```user_id.tsv: [user_name, '\t', ,user_id]```, user node id; (user_id should start from 0)

```item_id.tsv: [item_name, '\t', ,item_id]```, item node id; (item_id should start from 0)

```adjlist_user_id.tsv: [user_name, '\t', ,user id for adjlist]```, user node ids of the adjacency list file; (adjlist_user_id should start from 0, which is exactly the same as user_id.)

```adjlist_item_id.tsv: [item_name, '\t', ,item_id for adjlist]```, item node ids of the adjacency list file; (It's suggested that the adjlist_item_id should start from the end of adjlist_user_id. For instance, if adjlist_user_id is from 0 to 100, the adjlist_item_id should start from 101.)

```adjlist.txt: [node_itself neighbor_node_0 neighbor_node_1 nerighbor_node_2 neighbor_node_3 ... neighbor_node_k]```, the adjacency list for the graph (training set), each node is represented as its adjlist id;

```train.csv: [user_id, item_id]```, the dataset for embedding model training. It only contains true links of the inter-partition relations. We take them as positive cases and randomly sample negative cases during the training process to model the inter-partition proximity;

```valid.tsv: [user_id, '\t', item_id, '\t', label]```, the dataset for embedding model validation. It contains both positive cases and negative cases (randomly sampled) for inter-partition links. ```label``` indicates that whether the link relation is positive or not. The ratio of positives to negatives is 1:1;

```train.tsv: [user_id, '\t', item_id, '\t', label]```, the dataset for training link prediction model (a logistic regression model). The ```label``` information and positive to negative ratio is the same to ```valid.tsv```;

```test.tsv: [user_id, '\t', tem_id, '\t', label]```, the test set for link prediction. The ```label``` information and positive to negative ratio is the same to ```valid.tsv```;

```user_attr.pkl: user_attr[user_id][:] ```, a two-dimension ndarray matrix of user attributes, the row number (starts from 0) represents the respective user id.

```item_attr.pkl: item_attr[item_id][:]```, a two-dimension ndarray matrix of item attributes, the row number (starts from 0) represents the respective item id.

```
emb.txt:
        node_number, dimension (skip this line)
        <\s>(invalid token), embedding (skip this line)
        node_adjlist_id, embedding
        ......                   
```
, a matrix of high-order structure features for nodes. Each node is adjlist id. 
This file is the output of [metapath2vec++](https://ericdongyx.github.io/metapath2vec/m2v.html).

### Note
The above described dataset format is only required for the running python scripts in ```model```. One can change the dataset format as his/her wish if he/she considers to modify the data reading/writing code in ```model```.

## Training Output
```{dataset}_best_model.pkl```, the parameters of the trained AutoEncoders.


## Usage
### Requirements.
nmslib 2.0.1+ (please refer to [Non-Metric Space Library (NMSLIB)](https://github.com/nmslib/nmslib) for HNSW installation)  
tensorflow 1.10-1.15

### Model Training
- AMiner:
  ```
  cd model
  python gen_metapath.py --dataset ami --path_per_node 10 --path_length 81
  ./code_metapath2vec/metapath2vec -train ../data/ami/metapath_ami.txt -output ../data/ami/emb_ami -pp 1 -size 128 -window 3 -negative 5 -threads 32
  python train.py --dataset ami
  ```
- MovieLens
  ```
  cd model
  python gen_metapath.py --dataset mvl --path_per_node 10 --path_length 81
  ./code_metapath2vec/metapath2vec -train ../data/mvl/metapath_mvl.txt -output ../data/mvl/emb_mvl -pp 1 -size 128 -window 3 -negative 5 -threads 32
  python train.py --dataset mvl --lambda_6 10 --lambda_9 10 --attr_dim_0_u 23 --attr_dim_0_v 18 --attr_dim_1 32 --attr_dim_2 64 --struc_dim_1 96 --struc_dim_2 64
  ```

### Link Prediction
- AMiner:
  ```
  python link_prediction.py --dataset ami
  ```
- MovieLens
  ```
  python link_prediction.py --dataset mvl --attr_dim_0_u 23 --attr_dim_0_v 18 --attr_dim_1 32 --attr_dim_2 64 --struc_dim_1 96 --struc_dim_2 64
  ```
  
## Further Support
If you have any enquiries, please contact huangwentao@u.nus.edu (Huang Wentao) for the further support.  


## Bibliography
```
@inproceedings{DBLP:conf/sigir/HuangL0FY20,
  author       = {Wentao Huang and
                  Yuchen Li and
                  Yuan Fang and
                  Ju Fan and
                  Hongxia Yang},
  title        = {BiANE: Bipartite Attributed Network Embedding},
  booktitle    = {{SIGIR}},
  pages        = {149--158},
  publisher    = {{ACM}},
  year         = {2020}
}
```
