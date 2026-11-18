import numpy as np
import torch
from torch import nn

from src.ioi_data import make_examples, generate_splits, NAMES
from src.sae import TopKSAE, train_sae
from src.experiment import (
    select_feature, heldout_auc, feature_ablation_deltas,
    collect_activations, gaps_and_accuracy, matched_controls,
)


def test_balanced_data_and_uniqueness():
    rows = make_examples(['John', 'Mary', 'James'],
                         ['When {first} and {second} met, {subject} gave a book to'],
                         10, seed=3)
    assert len(rows) == 10
    assert len({x['prompt'] for x in rows}) == 10
    assert sum(x['io_first'] for x in rows) == 5
    assert all(x['subject'] != x['io'] for x in rows)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {'[PAD]': 0}
        self.padding_side = 'right'

    def encode(self, text, add_special_tokens=False):
        return [self._id(tok) for tok in text.strip().split()]

    def _id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def __call__(self, texts, padding=True, return_tensors='pt'):
        tokens = [self.encode(t) for t in texts]
        max_len = max(map(len, tokens))
        return {
            'input_ids': torch.tensor([t + [0] * (max_len - len(t)) for t in tokens]),
            'attention_mask': torch.tensor([[1] * len(t) + [0] * (max_len - len(t)) for t in tokens]),
        }


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Linear(d, d)

    def forward(self, hidden):
        return (self.mlp(hidden).tanh(),)


class FakeTransformer(nn.Module):
    def __init__(self, n_tokens=200, d=12):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d)
        self.h = nn.ModuleList([FakeBlock(d)])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.h[0](self.embed(input_ids))[0]


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakeTransformer()
        self.head = nn.Linear(12, 200)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return type('Output', (), {'logits': self.head(self.transformer(input_ids, attention_mask))})()


def test_hook_and_ablation_pipeline():
    torch.manual_seed(1)
    tokenizer = FakeTokenizer()
    model = FakeLM().eval()
    rows = make_examples(['John','Mary','James','Sarah'],
        ['When {first} and {second} met, {subject} gave a book to'], 8, seed=1)
    acts = collect_activations(model, tokenizer, rows, layer=0, batch_size=3)
    assert acts.shape == (8, 12)
    sae = TopKSAE(12, 32, k=4)
    sae.fit_normalization(acts)
    delta = feature_ablation_deltas(sae, acts, feature=1, device='cpu')
    assert delta.shape == (8, 12)
    gaps, acc = gaps_and_accuracy(model, tokenizer, rows, batch_size=3)
    patched_gaps, _ = gaps_and_accuracy(model, tokenizer, rows, batch_size=3, layer=0, deltas=delta)
    assert len(gaps) == 8
    assert 0 <= acc <= 1
    assert np.isfinite(patched_gaps).all()
    # Each call should remove its hooks when it finishes.
    assert len(model.transformer.h[0]._forward_hooks) == 0


def test_sae_fit_and_metrics():
    torch.manual_seed(10)
    train = torch.randn(100, 12) * 2 + 10
    dev = torch.randn(30, 12) * 2 + 10
    sae, history = train_sae(train, dev, d_sae=36, k=5, steps=12,
                            batch_size=16, device='cpu', log_every=6)
    assert len(history) >= 2
    metrics = sae.reconstruction_metrics(dev)
    assert np.isfinite(metrics['mse'])
    assert metrics['l0'] <= 5
    assert sae.mean.shape == (12,)


def test_feature_discovery_has_no_test_leakage():
    rng = np.random.default_rng(0)
    rows = [{'io_first': i % 2} for i in range(100)]
    z = rng.random((100, 8)) * (rng.random((100, 8)) > .5) * .02
    z[:,2] = np.array([x['io_first'] for x in rows]) * 3
    selected, sign, score, freq = select_feature(z, rows)
    assert selected == 2 and sign == 1
    assert heldout_auc(z, rows, selected, sign) > .98
    controls = matched_controls(z, selected, score, freq, n=3)
    assert selected not in controls


def test_split_names_are_disjoint_and_balanced(tmp_path):
    tok = FakeTokenizer()
    pools = generate_splits(tok, tmp_path, n_train=80, n_dev=32, n_test=32, n_ood=16)
    assert not (set(pools['train']) & set(pools['dev']))
    assert not (set(pools['train']) & set(pools['test']))
    assert not (set(pools['dev']) & set(pools['test']))
    from src.ioi_data import load_examples
    for split, n in [('train',80), ('dev',32), ('test',32), ('ood',16)]:
        rows = load_examples(tmp_path / f'{split}.json')
        assert len(rows) == n
        assert sum(e['io_first'] for e in rows) == n // 2
