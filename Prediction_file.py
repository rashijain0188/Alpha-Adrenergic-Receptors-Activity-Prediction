# -*- coding: utf-8 -*-
"""
dmpnn_predict.py -
========================================================================================
    python dmpnn_predict_standalone.py --model_dir <run_dir>/model --input mols.csv --smiles_col SMILES --threshold 0.5
"""
import os, json, pickle, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool

ENCODER_PKL = "shared_encoder.pkl"
HEADS_PKL   = "task_heads.pkl"
MIN_PARALLEL = 64          
BATCH_SIZE   = 64

_HYB = [Chem.rdchem.HybridizationType.S,   Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
_CHI = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER]
_BT  = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
_BST = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,    Chem.rdchem.BondStereo.STEREOE]

_ATOMIC_NUM = list(range(1, 119))
_DEGREE     = list(range(0, 11))
_CHARGE     = [-2, -1, 0, 1, 2, 3]
_VALENCE    = list(range(0, 9))
_NUM_HS     = list(range(0, 9))
_BOOL       = [False, True]

def _one_hot(value, choices):
    enc = [0] * (len(choices) + 1)
    enc[choices.index(value) if value in choices else len(choices)] = 1
    return enc

def _atom_features(a):
    f  = _one_hot(a.GetAtomicNum(),     _ATOMIC_NUM)
    f += _one_hot(a.GetDegree(),        _DEGREE)
    f += _one_hot(a.GetFormalCharge(),  _CHARGE)
    f += _one_hot(a.GetTotalValence(),  _VALENCE)
    f += _one_hot(a.GetHybridization(), _HYB)
    f += _one_hot(a.GetTotalNumHs(),    _NUM_HS)
    f += _one_hot(a.IsInRing(),         _BOOL)
    f += _one_hot(a.GetIsAromatic(),    _BOOL)
    f += [a.GetMass() / 100.0, 0.0]
    f += _one_hot(a.GetChiralTag(),     _CHI)
    cip = a.GetPropsAsDict().get("_CIPCode", "")
    f += [int(cip == "R"), int(cip == "S")]
    return f

def _bond_features(b):
    f  = _one_hot(b.GetBondType(),     _BT)
    f += _one_hot(b.GetIsConjugated(), _BOOL)
    f += _one_hot(b.IsInRing(),        _BOOL)
    f += _one_hot(b.GetStereo(),       _BST)
    return f

class DMPNNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "reverse_edge_index":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None, None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, attr, rev = [], [], [], []
    for k, bond in enumerate(mol.GetBonds()):
        i, j, bf = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _bond_features(bond)
        src += [i, j]; dst += [j, i]; attr += [bf, bf]
        rev += [2 * k + 1, 2 * k]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(attr, dtype=torch.float)
        rev_index  = torch.tensor(rev, dtype=torch.long)
    else:                                           
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 16), dtype=torch.float)
        rev_index  = torch.zeros((0,), dtype=torch.long)
    return canonical, DMPNNData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                reverse_edge_index=rev_index, n_atoms=x.shape[0])

class DMPNNEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden=300, n_layers=3, dropout=0.0):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.W_i = nn.Sequential(nn.Linear(node_in + edge_in, hidden, bias=False),
                                 nn.BatchNorm1d(hidden))
        self.W_m = nn.Linear(hidden, hidden, bias=False)
        self.W_m_norm = nn.LayerNorm(hidden)
        self.W_o = nn.Sequential(nn.Linear(node_in + hidden, hidden),
                                 nn.LayerNorm(hidden), nn.SiLU())
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        rev, batch = data.reverse_edge_index, data.batch
        src, dst = edge_index[0], edge_index[1]
        n_atoms = x.shape[0]
        expand = lambda idx: idx.unsqueeze(1).expand(-1, self.hidden)

        h = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        for _ in range(self.n_layers):
            agg = torch.zeros(n_atoms, self.hidden, device=x.device)
            agg = agg.scatter_add(0, expand(dst), h)
            m = agg[src] - h[rev]
            h = self.dropout(self.act(self.W_m_norm(self.W_m(m)) + h.clone()))

        atom_msg = torch.zeros(n_atoms, self.hidden, device=x.device)
        atom_msg = atom_msg.scatter_add(0, expand(src), h)
        h_atoms = self.W_o(torch.cat([x, atom_msg], dim=-1))
        return torch.cat([global_mean_pool(h_atoms, batch),
                          global_add_pool(h_atoms, batch)], dim=-1)

def _make_head(hidden, n_classes=1):
    return nn.Sequential(
        nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.0),
        nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.SiLU(), nn.Dropout(0.0),
        nn.Linear(hidden // 2, hidden // 4), nn.SiLU(),
        nn.Linear(hidden // 4, n_classes))

class DMPNNPredictor(nn.Module):
    def __init__(self, enc_cfg, n_tasks):
        super().__init__()
        self.encoder = DMPNNEncoder(enc_cfg["node_in"], enc_cfg["edge_in"],
                                    enc_cfg["hidden"], enc_cfg["n_layers"],
                                    enc_cfg.get("dropout", 0.0))
        self.task_heads = nn.ModuleList(_make_head(enc_cfg["hidden"]) for _ in range(n_tasks))

    def forward(self, data):
        g = self.encoder(data)
        return torch.stack([h(g).squeeze(-1) for h in self.task_heads], dim=1)


def load_model(model_dir):
    with open(os.path.join(model_dir, ENCODER_PKL), "rb") as f:
        enc = pickle.load(f)
    with open(os.path.join(model_dir, HEADS_PKL), "rb") as f:
        heads = pickle.load(f)
    if heads.get("output") != "sigmoid":
        raise ValueError("Error!")

    model = DMPNNPredictor(enc["config"], heads["n_tasks"])
    model.encoder.load_state_dict(enc["state_dict"])
    model.task_heads.load_state_dict(heads["state_dict"])
    model.eval().requires_grad_(False)

    return model, list(heads["task_names"])

_MODEL = {}

def _init_worker(model_dir):
    torch.set_num_threads(1)
    _MODEL["m"] = load_model(model_dir)[0]

@torch.no_grad()
def _predict_chunk(smiles_list, model=None):
    model = model or _MODEL["m"]
    n_tasks = len(model.task_heads)
    canonical = [None] * len(smiles_list)
    probs = np.full((len(smiles_list), n_tasks), np.nan)

    rows, graphs = [], []
    for i, s in enumerate(smiles_list):
        if not isinstance(s, str) or not s.strip():
            continue
        try:
            can, g = smiles_to_graph(s.strip())
        except Exception:
            can, g = None, None
        if g is not None:
            canonical[i] = can
            rows.append(i)
            graphs.append(g)

    pos = 0
    for batch in DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False):
        n = batch.num_graphs
        probs[rows[pos:pos + n]] = torch.sigmoid(model(batch)).numpy()
        pos += n
    return canonical, probs

def _chunk_job(chunk):
    return _predict_chunk(chunk)

def predict_smiles(model_dir, model, smiles, cores=1):
    smiles = list(smiles)
    if cores > 1 and len(smiles) >= MIN_PARALLEL:
        size = max(1, -(-len(smiles) // (cores * 4)))
        chunks = [smiles[i:i + size] for i in range(0, len(smiles), size)]
        with mp.get_context("spawn").Pool(min(cores, len(chunks)), _init_worker,
                                          (model_dir,)) as pool:
            res = pool.map(_chunk_job, chunks, chunksize=1)
        return ([c for r in res for c in r[0]], np.concatenate([r[1] for r in res], 0))
    return _predict_chunk(smiles, model)

def main():
    p = argparse.ArgumentParser(description="Predict P(active) from SMILES CSV files.")
    p.add_argument("--model_dir", required=True,
                   help="folder containing shared_encoder.pkl and task_heads.pkl")
    p.add_argument("--input", nargs="+", required=True, help="one or more CSV files")
    p.add_argument("--smiles_col", default="SMILES", help="name of the SMILES column")
    p.add_argument("--output_dir", default="predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cores", type=int, default=1, help="worker processes (1 = serial)")
    a = p.parse_args()

    model, task_names = load_model(a.model_dir)
    os.makedirs(a.output_dir, exist_ok=True)
    print(f"Loaded model from {a.model_dir}  |  tasks: {task_names}")

    for path in a.input:
        df = pd.read_csv(path)
        col = a.smiles_col
        if col not in df.columns:
            raise KeyError("SMILES column not found")

        canonical, probs = predict_smiles(a.model_dir, model, df[col].tolist(),
                                          max(1, a.cores))
        out = df.copy()
        out["canonical_smiles"] = canonical
        out["valid"] = [c is not None for c in canonical]
        for t, name in enumerate(task_names):
            out[f"prob_{name}"] = probs[:, t]
            pred = np.where(np.isnan(probs[:, t]), np.nan,
                            (probs[:, t] >= a.threshold).astype(float))
            out[f"pred_{name}"] = pd.Series(pred, index=out.index).astype("Int64")

        out_path = os.path.join(a.output_dir,
                                f"{os.path.splitext(os.path.basename(path))[0]}_predictions.csv")
        out.to_csv(out_path, index=False)
        n_bad = int((~out["valid"]).sum())
        print(f"{path}: {len(out)} rows"
              + (f", {n_bad} invalid SMILES" if n_bad else "") + f"  ->  {out_path}")

if __name__ == "__main__":     
    main()