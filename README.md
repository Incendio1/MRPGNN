# MRPGNN: A Decoupled Graph Neural Network Based on Multi-Hop Random Propagation
## Usage Example
### Cora (homophilic graph)
```javascript
python trdgnn.py --dataset=Cora --K=10 --dropout=0.85 --dropnode_rate=0.15
```

### Cornell (heterophilic graph)
```javascript
python trdgnn.py --dataset=cornell --dropout=0.5 --alpha=0.35 --beta=0.2 --dropnode_rate=0.15
```


