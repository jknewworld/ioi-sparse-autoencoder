"""Train a Top-K sparse autoencoder on values saved from a fixed GPT-2 model."""
import torch
from torch import nn
from torch.nn import functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k=32):
        super().__init__()
        if not (0 < k <= d_sae):
            raise ValueError('k must be in [1, d_sae].')
        self.d_in, self.d_sae, self.k = int(d_in), int(d_sae), int(k)
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder_weight = nn.Parameter(torch.randn(d_sae, d_in))
        self.decoder_bias = nn.Parameter(torch.zeros(d_in))
        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder_weight.copy_(F.normalize(self.decoder_weight, dim=1))
        self.register_buffer('mean', torch.zeros(d_in))
        self.register_buffer('scale', torch.ones(()))

    def fit_normalization(self, activations):
        with torch.no_grad():
            mean = activations.mean(dim=0)
            scale = (activations - mean).square().mean().sqrt().clamp_min(1e-5)
            self.mean.copy_(mean)
            self.scale.copy_(scale)

    def normalize(self, x):
        return (x - self.mean) / self.scale

    def encode(self, x):
        v = self.encoder(x)
        vals, idx = v.topk(self.k, dim=-1)
        z = torch.zeros_like(v)
        z.scatter_(dim=-1, index=idx, src=vals.clamp_min(0))
        return z

    def decode(self, z):
        return z @ F.normalize(self.decoder_weight, dim=1) + self.decoder_bias

    def forward(self, normalized_x):
        z = self.encode(normalized_x)
        return self.decode(z), z

    @torch.no_grad()
    def reconstruction_metrics(self, raw_x):
        x = self.normalize(raw_x)
        rec, z = self(x)
        mse = (rec - x).square().mean().item()
        denom = (x - x.mean(dim=0)).square().mean().clamp_min(1e-8)
        explained = 1 - ((rec - x).square().mean() / denom).item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        dead_fraction = (z.sum(dim=0) == 0).float().mean().item()
        return dict(mse=mse, explained_variance=explained, l0=l0, dead_feature_fraction=dead_fraction)


def train_sae(train_raw, dev_raw, *, d_sae=1536, k=32, steps=800, batch_size=256,
              lr=3e-4, seed=42, device='cpu', log_every=100):
    torch.manual_seed(seed)
    d_in = train_raw.shape[1]
    sae = TopKSAE(d_in, d_sae, k).to(device)
    sae.fit_normalization(train_raw.to(device))
    train = sae.normalize(train_raw.to(device))
    dev_raw = dev_raw.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None
    history = []
    for step in range(1, steps + 1):
        sae.train()
        indices = torch.randint(0, len(train), (batch_size,), device=device)
        batch = train[indices]
        rec, z = sae(batch)
        loss = (rec - batch).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            sae.eval()
            m = sae.reconstruction_metrics(dev_raw)
            history.append({'step': step, 'train_mse': float(loss.detach()), **m})
            print(f'[SAE] step {step:4d}: train_MSE={loss.item():.4f} '
                  f'dev_MSE={m["mse"]:.4f} explained={m["explained_variance"]:.3f} L0={m["l0"]:.1f}')
            if m['mse'] < best_loss:
                best_loss = m['mse']
                best_state = {key: val.detach().cpu().clone() for key, val in sae.state_dict().items()}
    if best_state is not None:
        sae.load_state_dict(best_state)
    sae.eval()
    return sae, history
