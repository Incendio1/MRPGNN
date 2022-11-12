# MRPGNN: A Decoupled Graph Neural Network Based on Multi-Hop Random Propagation
## Usage Example
### Cora (homophilic graph)
```javascript
python trdgnn.py --dataset=Cora --K=10 --dropout=0.85 --dropnode_rate=0.15
```
### Cornell (heterophilic graph)
python trdgnn.py --dataset=cornell --dropout=0.5 --alpha=0.35 --beta=0.2 --dropnode_rate=0.15

## Results
model |Cora |CiteSeer |PubMed|Amazon Computers |Amazon Photo |Coauthor CS
------ | -----  |----------- |---|--- | -----  |----------- |
NE-WNA| 82.8% | 74.2%| 82.5%|84.7%| 93.2% | 92.5%|


