"""Make IOI prompts, train an SAE, then test what changes when a feature is removed."""
import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

from src.ioi_data import generate_splits, load_examples
from src.sae import TopKSAE, train_sae
from src.experiment import (
    collect_activations, gaps_and_accuracy, feature_activations,
    feature_ablation_deltas, select_feature, heldout_auc,
    matched_controls, bootstrap_ci,
)

MODEL_ID = 'openai-community/gpt2'


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def checkpoint_load(path, device):
    # Load the SAE weights saved by the training step.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    sae = TopKSAE(cfg['d_in'], cfg['d_sae'], cfg['k']).to(device)
    sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()
    return sae, cfg


def get_acts(model, tokenizer, data_dir, output_dir, split, layer, batch, reuse_cache=True):
    path = output_dir / f'acts_{split}_layer{layer}.pt'
    rows = load_examples(data_dir / f'{split}.json')
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode('utf8')).hexdigest()
    if reuse_cache and path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(cache, dict) and cache.get('data_sha256') == fingerprint and cache.get('model') == MODEL_ID:
            acts = cache['activations']
            if acts.shape == (len(rows), model.config.n_embd):
                return acts
    print(f'[Activations] {split}: {len(rows)} examples')
    acts = collect_activations(model, tokenizer, rows, layer=layer, batch_size=batch)
    torch.save({'activations': acts, 'data_sha256': fingerprint, 'model': MODEL_ID}, path)
    return acts


def group_stats(changes, rows):
    y = np.array([r['io_first'] for r in rows])
    return {
        'mean_gap_change': float(changes.mean()),
        'mean_absolute_gap_change': float(np.abs(changes).mean()),
        'rms_gap_change': float(np.sqrt(np.mean(changes ** 2))),
        'mean_gap_change_io_first': float(changes[y == 1].mean()),
        'mean_gap_change_io_second': float(changes[y == 0].mean()),
        'mean_gap_change_95pct_bootstrap_ci': bootstrap_ci(changes),
    }


def evaluate_split(model, tokenizer, sae, rows, acts, layer, batch, chosen,
                   controls, sign, output_dir, split_name, device):
    print(f'[Evaluate] {split_name}: original logits')
    baseline_gap, baseline_acc = gaps_and_accuracy(model, tokenizer, rows, batch_size=batch)
    z = feature_activations(sae, acts, device)
    result = {
        'n': len(rows),
        'baseline_mean_logit_gap': float(baseline_gap.mean()),
        'baseline_pairwise_accuracy': float(baseline_acc),
        'sae_reconstruction': sae.reconstruction_metrics(acts.to(device)),
        'feature_order_auc': heldout_auc(z, rows, chosen, sign),
        'selected_active_fraction': float((z[:, chosen] > 0).mean()),
    }
    changes = {}
    for i, feature in enumerate([chosen] + controls):
        print(f'[Evaluate] {split_name}: remove feature {feature} ({i + 1}/{len(controls) + 1})')
        delta = feature_ablation_deltas(sae, acts, feature, device)
        patched_gap, patched_acc = gaps_and_accuracy(
            model, tokenizer, rows, batch_size=batch, layer=layer, deltas=delta,
        )
        change = patched_gap - baseline_gap
        changes[int(feature)] = change
        result_key = 'selected_feature' if i == 0 else f'control_{i - 1}'
        result[result_key] = {
            'feature_id': int(feature),
            'pairwise_accuracy_after_ablation': float(patched_acc),
            **group_stats(change, rows),
        }
    control_effects = [result[f'control_{i}']['rms_gap_change'] for i in range(len(controls))]
    result['control_median_rms_gap_change'] = float(np.median(control_effects)) if control_effects else None
    result['selected_rms_exceeds_all_controls'] = bool(
        result['selected_feature']['rms_gap_change'] > max(control_effects)) if control_effects else None

    with (output_dir / f'{split_name}_examples.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['prompt', 'correct_io', 'subject', 'io_first', 'original_logit_gap',
                    'selected_feature_ablation_gap_change'])
        for r, gap, change in zip(rows, baseline_gap, changes[chosen]):
            w.writerow([r['prompt'], r['io'], r['subject'], r['io_first'], round(float(gap), 5),
                        round(float(change), 5)])
    return result, z


def make_plots(report, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train_hist = report.get('train_history', [])
    if train_hist:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([i['step'] for i in train_hist], [i['mse'] for i in train_hist], marker='o')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Mean squared error on the scaled development data')
        ax.set_title('How training changes the SAE error')
        fig.tight_layout()
        fig.savefig(output_dir / 'sae_loss.png', dpi=170)
        plt.close(fig)
    if 'test' in report:
        r = report['test']
        entries = [r['selected_feature']] + [r[k] for k in sorted((k for k in r if k.startswith('control_') and k[8:].isdigit()), key=lambda k: int(k[8:]))]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['selected'] + [f'control {i + 1}' for i in range(len(entries) - 1)],
               [e['rms_gap_change'] for e in entries])
        ax.set_ylabel('Size of the score change (RMS, logits)')
        ax.set_title('Removing the chosen feature and the comparison features')
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout()
        fig.savefig(output_dir / 'causal_controls.png', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['data', 'baseline', 'train', 'evaluate', 'all'], default='all')
    parser.add_argument('--out', default='results')
    parser.add_argument('--layer', type=int, default=8, help='Save values after this GPT-2 block. Block numbers start at zero.')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--d-sae', type=int, default=1536)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--batch', type=int, default=16, help='Number of prompts GPT-2 reads at once')
    parser.add_argument('--controls', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--quick', action='store_true', help='Use a small run to check that the code works')
    args = parser.parse_args()
    if args.quick:
        args.steps = 15
        args.d_sae = 128
        args.k = 8
        args.controls = 2
    if not 0 <= args.layer < 12:
        parser.error('--layer must be 0..11 for GPT-2 small')
    out = Path(args.out)
    data_dir = out / 'data'
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    device = 'cpu' if device == 'auto' else device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import here so the data and SAE tests can run without transformers.
    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as ex:
        raise SystemExit('Install dependencies first: pip install -r requirements.txt') from ex
    print(f'Model={MODEL_ID}, device={device}, stage={args.stage}, output={out}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    sizes = dict(n_train=256, n_dev=64, n_test=64, n_ood=32) if args.quick else dict(
        n_train=2048, n_dev=160, n_test=160, n_ood=80)
    if args.stage in ('data', 'all'):
        name_pools = generate_splits(tokenizer, data_dir, seed=args.seed, **sizes)
        print('[Data] Number of names in the train, dev, and test sets:', {k:len(v) for k,v in name_pools.items()})
        if args.stage == 'data':
            return
    if not (data_dir / 'train.json').exists():
        raise SystemExit('First run: python run_pipeline.py --stage data')
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(device).eval()
    if args.stage in ('baseline', 'all'):
        print('[Baseline] comparing the two name scores on the development set')
        dev_rows = load_examples(data_dir / 'dev.json')
        gap, acc = gaps_and_accuracy(model, tokenizer, dev_rows, batch_size=args.batch)
        m = {'n':len(dev_rows), 'mean_gap':float(gap.mean()), 'pairwise_accuracy':float(acc)}
        dump_json(out / 'baseline_dev.json', m)
        print('[Baseline] dev:', m)
        if args.stage == 'baseline':
            return
    sae_path = out / f'sae_layer{args.layer}.pt'
    train_manifest_hash = hashlib.sha256((data_dir / 'train.json').read_bytes()).hexdigest()
    if args.stage in ('train', 'all'):
        train_act = get_acts(model, tokenizer, data_dir, out, 'train', args.layer, args.batch)
        dev_act = get_acts(model, tokenizer, data_dir, out, 'dev', args.layer, args.batch)
        sae, history = train_sae(
            train_act, dev_act, d_sae=args.d_sae, k=args.k, steps=args.steps,
            device=device, seed=args.seed,
        )
        cfg = {'model':MODEL_ID, 'layer':args.layer, 'd_in':int(train_act.shape[1]),
               'd_sae':args.d_sae, 'k':args.k, 'steps':args.steps, 'seed':args.seed,
               'site':'GPT2 transformer.h[layer] output, final non-padding token',
               'data_size':len(train_act), 'quick':args.quick,
               'train_file_sha256':train_manifest_hash}
        torch.save({'config':cfg, 'state_dict':sae.cpu().state_dict()}, sae_path)
        dump_json(out / 'sae_train_history.json', history)
        print('[Train] checkpoint saved:', sae_path)
        if args.stage == 'train':
            return
    if not sae_path.exists():
        raise SystemExit('First run: python run_pipeline.py --stage train')
    sae, cfg = checkpoint_load(sae_path, device)
    if cfg['layer'] != args.layer:
        raise SystemExit('Selected SAE checkpoint is for a different layer.')
    if cfg.get('train_file_sha256') != train_manifest_hash:
        raise SystemExit('The training data changed after the SAE was saved. Run --stage train again.')
    if cfg.get('quick'):
        print('Quick mode uses a small set of prompts to check the code.')
    dev_rows = load_examples(data_dir / 'dev.json')
    dev_act = get_acts(model, tokenizer, data_dir, out, 'dev', args.layer, args.batch)
    dev_z = feature_activations(sae, dev_act, device)
    selected, sign, scores, freq = select_feature(dev_z, dev_rows)
    controls = matched_controls(dev_z, selected, scores, freq, n=args.controls, seed=args.seed)
    print('[Discovery] feature:', selected, 'sign:', sign, 'dev_AUC:', heldout_auc(dev_z, dev_rows, selected, sign))
    print('[Controls] IDs:', controls)
    report = {
        'model': MODEL_ID,
        'environment': {'python':platform.python_version(), 'torch':torch.__version__,
                        'transformers':__import__('transformers').__version__, 'device':device},
        'activation_site': f'layer {args.layer} resid-post (final prompt token)',
        'sae_config': cfg,
        'selected_feature_id':int(selected),
        'selected_feature_direction':int(sign),
        'dev_auc':heldout_auc(dev_z, dev_rows, selected, sign),
        'controls':controls,
        'train_history':json.loads((out / 'sae_train_history.json').read_text()) if (out / 'sae_train_history.json').exists() else [],
        'note':'We chose the feature using only the development prompts.',
        'limitation':'This experiment tests how one feature responds to name order and how removing it changes the scores.'
    }
    for split in ('test', 'ood'):
        rows = load_examples(data_dir / f'{split}.json')
        acts = get_acts(model, tokenizer, data_dir, out, split, args.layer, args.batch)
        report[split], z = evaluate_split(
            model, tokenizer, sae, rows, acts, args.layer, args.batch, selected,
            controls, sign, out, split, device,
        )
    # Train a simple classifier on all the GPT-2 values to predict name order.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    y_dev = np.asarray([e['io_first'] for e in dev_rows])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, C=0.1))
    probe.fit(dev_act.numpy(), y_dev)
    for split in ('test', 'ood'):
        rows = load_examples(data_dir / f'{split}.json')
        acts = get_acts(model, tokenizer, data_dir, out, split, args.layer, args.batch)
        y = [e['io_first'] for e in rows]
        report[split]['linear_probe_auc'] = float(roc_auc_score(y, probe.predict_proba(acts.numpy())[:,1]))
    dump_json(out / 'summary.json', report)
    make_plots(report, out)
    print('\nComplete. Read:', out / 'summary.json')
    print('Charts:', out / 'sae_loss.png', out / 'causal_controls.png')


if __name__ == '__main__':
    main()
