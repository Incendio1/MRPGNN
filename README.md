# MRPGNN: A Decoupled Graph Neural Network Based on Multi-Hop Random Propagation
## Usage Example
### Cora (homophilic graph)
```javascript
python mrpgnn.py --dataset=Cora --K=10 --dropout=0.85 --dropnode_rate=0.15
```

### Cornell (heterophilic graph)
```javascript
python mrpgnn.py --dataset=cornell --dropout=0.5 --alpha=0.35 --beta=0.2 --dropnode_rate=0.15
```
## Results
model	|Cora	|CiteSeer	|PubMed|Chameleon|Actor	|Texas	|Cornell
------ | -----  |----------- |---|--- | -----  |----------- |-------
MRPGNN|	85.2% |	74.0%|	80.9%|62.3%|	40.0% |	93.6%|93.1%


