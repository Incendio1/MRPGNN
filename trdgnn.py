import argparse
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from train_eval import *
from datasets import *
import warnings


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--dropnode_rate', type=float, default=0.15)
parser.add_argument('--dprate', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.02)
parser.add_argument('--beta', type=float, default=0.1)

args = parser.parse_args()


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def rand_prop(features, training):  # Mask_Node
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        features = masks.cuda() * features

    else:

        features = features * (1. - drop_rate)
    # features = propagate(features, A, args.order)
    return features


class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        preds = []
        preds.append(x)
        z = preds[0]
        alpha = args.alpha
        beta = args.beta
        for k in range(self.K):
            x = rand_prop(x, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm)
            if k == 0:
                x = (1-alpha) * self.propagate(edge_index, x=x, norm=norm) + alpha * z
            else:
                x = (1-alpha) * self.propagate(edge_index, x=x, norm=norm) + alpha * z + beta * preds[-1]
            preds.append(x)

        # # other adaptive integration
        # out = torch.zeros_like(preds[0])
        # U = sum(preds)
        # U = U.mean(-1).mean(-1)
        # score_list = []
        # for i in range(self.K):
        #     score = torch.sigmoid(U * preds[i])
        #     score_list.append(score)
        # for i in range(len(score_list)):
        #     out += score[i] * preds[i]
        #
        # return out

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = Prop(dataset.num_classes, args.K)
        self.dprate = args.dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


warnings.filterwarnings("ignore", category=UserWarning)

if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset[0])
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks,
        lcc=False)
elif args.dataset == "cs" or args.dataset == "physics":
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks,
        lcc=False)
elif args.dataset == "computers" or args.dataset == "photo":
    dataset = get_amazon_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks,
        lcc=True)
elif args.dataset == "texas" or args.dataset == "cornell" or args.dataset == 'wisconsin':
    dataset = WebKB(root='../data/',
                    name=args.dataset, transform=T.NormalizeFeatures())
    print("Data:", dataset[0])
    dname = args.dataset
    dataset, data = DataLoader(dname)
    runs = args.runs
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)
    Results0 = []
    for RP in tqdm(range(runs)):
        test_acc, best_val_acc = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc])

    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'DAGNN on dataset {args.dataset}, in {runs} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')

elif args.dataset == "chameleon" or args.dataset == "squirrel":
    # use everything from "geom_gcn_preprocess=False" and
    # only the node label y from "geom_gcn_preprocess=True"
    preProcDs = WikipediaNetwork(
        root='../data/', name=args.dataset, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
    dataset = WikipediaNetwork(
        root='../data/', name=args.dataset, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    dname = args.dataset
    dataset, data = DataLoader(dname)
    runs = args.runs
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)
    Results0 = []
    for RP in tqdm(range(runs)):
        test_acc, best_val_acc = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc])

    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'DAGNN on dataset {args.dataset}, in {runs} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
    # run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, lcc=True)
elif args.dataset == "film":
    dataset = Actor(
        root='../data/film', transform=T.NormalizeFeatures())

    dname = args.dataset
    dataset, data = DataLoader(dname)
    runs = args.runs
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)
    Results0 = []
    for RP in tqdm(range(runs)):
        test_acc, best_val_acc = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc])

    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'DAGNN on dataset {args.dataset}, in {runs} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')





