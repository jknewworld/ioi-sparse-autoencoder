"""Read values from GPT-2, remove a feature, and compare the scores for two names."""
import numpy as np
import torch
from .feature_stats import bootstrap_ci, heldout_auc, matched_controls, select_feature


def _model_device(model):
    return next(model.parameters()).device


def _get_hidden(outputs):
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def _replace_hidden(outputs, hidden):
    return (hidden,) + tuple(outputs[1:]) if isinstance(outputs, (tuple, list)) else hidden


def _token_batch(tokenizer, examples, device):
    tokenizer.padding_side = 'right'
    batch = tokenizer([e['prompt'] for e in examples], padding=True, return_tensors='pt')
    return {k: v.to(device) for k, v in batch.items()}


def _last_positions(inputs):
    return inputs['attention_mask'].sum(dim=-1).long() - 1


def _answer_ids(tokenizer, examples, device):
    io_ids = [tokenizer.encode(' ' + e['io'], add_special_tokens=False) for e in examples]
    subj_ids = [tokenizer.encode(' ' + e['subject'], add_special_tokens=False) for e in examples]
    if any(len(x) != 1 for x in io_ids + subj_ids):
        raise ValueError('Each name must give one GPT-2 token when a space comes before it.')
    return (torch.tensor([x[0] for x in io_ids], device=device),
            torch.tensor([x[0] for x in subj_ids], device=device))


@torch.inference_mode()
def collect_activations(model, tokenizer, examples, layer, batch_size=16):
    """Save the values after this block at the last real token of each prompt."""
    device = _model_device(model)
    acts = []
    block = model.transformer.h[layer]
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        inputs = _token_batch(tokenizer, rows, device)
        pos = _last_positions(inputs)
        captured = []
        def hook(_, __, outputs):
            out = _get_hidden(outputs)
            captured.append(out[torch.arange(len(rows), device=device), pos].detach().cpu())
        h = block.register_forward_hook(hook)
        try:
            model.transformer(**inputs, use_cache=False)
        finally:
            h.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one group of saved values from this block.')
        acts.append(captured[0])
    return torch.cat(acts, dim=0)


@torch.inference_mode()
def gaps_and_accuracy(model, tokenizer, examples, batch_size=16, layer=None, deltas=None):
    """Return the score gap between the recipient and subject. Apply an edit if given."""
    device = _model_device(model)
    gaps, correct = [], []
    block = model.transformer.h[layer] if layer is not None else None
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        inputs = _token_batch(tokenizer, rows, device)
        pos = _last_positions(inputs)
        io_ids, subj_ids = _answer_ids(tokenizer, rows, device)
        if deltas is not None:
            local_delta = deltas[start:start + len(rows)].to(device)
            def intervene(_, __, outputs):
                out = _get_hidden(outputs)
                patched = out.clone()
                patched[torch.arange(len(rows), device=device), pos] += local_delta.to(patched.dtype)
                return _replace_hidden(outputs, patched)
            h = block.register_forward_hook(intervene)
        else:
            h = None
        try:
            outputs = model(**inputs, use_cache=False)
        finally:
            if h is not None:
                h.remove()
        logits = outputs.logits[torch.arange(len(rows), device=device), pos].float()
        gap = logits.gather(-1, io_ids[:, None]).squeeze(-1) - logits.gather(-1, subj_ids[:, None]).squeeze(-1)
        # A positive gap means GPT-2 gave the recipient a higher score than the subject.
        gaps.extend(gap.cpu().tolist())
        correct.extend((gap > 0).cpu().tolist())
    return np.asarray(gaps, dtype=float), float(np.mean(correct))


@torch.inference_mode()
def feature_activations(sae, raw_activations, device):
    return sae.encode(sae.normalize(raw_activations.to(device))).cpu().numpy()


@torch.inference_mode()
def feature_ablation_deltas(sae, raw_activations, feature, device):
    """Find the change needed to remove one feature from the original GPT-2 values."""
    x = sae.normalize(raw_activations.to(device))
    z = sae.encode(x)
    deleted = torch.zeros_like(z)
    deleted[:, feature] = z[:, feature]
    # Each decoder row has length one. Subtract the part from this feature.
    delta_norm = -deleted @ torch.nn.functional.normalize(sae.decoder_weight, dim=1)
    return (delta_norm * sae.scale).detach().cpu()
