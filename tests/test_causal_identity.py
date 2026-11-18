import torch
from src.sae import TopKSAE
from src.experiment import feature_ablation_deltas


def test_decoder_difference_matches_manual_and_zero_feature():
    torch.manual_seed(10)
    sae = TopKSAE(d_in=6, d_sae=12, k=3)
    data = torch.randn(12, 6)
    sae.fit_normalization(data)
    for f in [0, 4, 8]:
        with torch.no_grad():
            x = sae.normalize(data)
            z = sae.encode(x)
            modified = z.clone()
            modified[:, f] = 0
            expected = sae.scale * (sae.decode(modified) - sae.decode(z))
            actual = feature_ablation_deltas(sae, data, f, 'cpu')
            assert torch.allclose(expected, actual, atol=1e-6)
