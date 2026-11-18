"""Make two-name prompts. Each set has equal numbers of both name orders."""
import json
import random
from pathlib import Path

NAMES = [
    'John', 'Mary', 'James', 'Sarah', 'David', 'Michael', 'Emma', 'Lucy',
    'Daniel', 'Anna', 'Robert', 'Alice', 'Paul', 'Mark', 'Jennifer', 'Thomas',
    'Richard', 'Susan', 'Jessica', 'Kevin', 'Emily', 'Laura', 'Peter', 'Simon',
    'Rachel', 'Elizabeth', 'George', 'Lisa', 'Charles', 'Andrew', 'Brian',
    'Karen', 'Philip', 'Jack', 'Julia', 'Henry', 'Nick', 'Claire', 'Amy',
    'Alex', 'Jason', 'Nathan', 'Kelly', 'Helen', 'Steve', 'Joseph', 'Matthew',
    'Olivia', 'Sophie', 'Patrick', 'Edward', 'William', 'Grace', 'Jacob',
    'Samuel', 'Martha', 'Arthur', 'Charlotte', 'Angela', 'Diana', 'Martin',
]

TRAIN_TEMPLATES = [
    'When {first} and {second} went to the store, {subject} gave a bottle to',
    'After {first} and {second} visited the park, {subject} handed a ball to',
    'When {first} and {second} arrived at school, {subject} passed a note to',
    'At the market, {first} met {second}, and {subject} gave a present to',
]
OOD_TEMPLATES = [
    'Yesterday {first} and {second} went hiking. Later, {subject} offered a sandwich to',
    'During the trip, {first} spoke with {second}, and {subject} passed the camera to',
]


def single_token_names(tokenizer):
    """Keep names that give one token when a space comes before them."""
    result = [name for name in NAMES if len(tokenizer.encode(' ' + name, add_special_tokens=False)) == 1]
    if len(result) < 36:
        raise ValueError(f'Only {len(result)} eligible single-token names (need >=36).')
    return result


def make_examples(names, templates, n, seed):
    """Make different prompts with the recipient first in half and second in the rest."""
    if n % 2:
        raise ValueError('Use an even number of prompts so both name orders have the same count.')
    rng = random.Random(seed)
    rows = []
    for io_first in (0, 1):
        pool = []
        for template_id, template in enumerate(templates):
            for subject in names:
                for io in names:
                    if io == subject:
                        continue
                    first = io if io_first else subject
                    second = subject if io_first else io
                    pool.append({
                        'prompt': template.format(first=first, second=second, subject=subject),
                        'subject': subject,
                        'io': io,
                        'io_first': io_first,
                        'template_id': template_id,
                    })
        if n // 2 > len(pool):
            raise ValueError(f'Not enough unique prompts: requested {n // 2}, available {len(pool)} per class')
        rows.extend(rng.sample(pool, n // 2))
    rng.shuffle(rows)
    assert len({row['prompt'] for row in rows}) == len(rows), 'Duplicate prompts produced.'
    assert sum(x['io_first'] for x in rows) == n // 2
    return rows


def generate_splits(tokenizer, out_dir, n_train=2048, n_dev=160, n_test=160, n_ood=80, seed=42):
    names = single_token_names(tokenizer)
    rng = random.Random(seed)
    rng.shuffle(names)
    pools = {'train': names[:20], 'dev': names[20:28], 'test': names[28:36]}
    configs = [
        ('train', pools['train'], TRAIN_TEMPLATES, n_train, seed + 1),
        ('dev', pools['dev'], TRAIN_TEMPLATES, n_dev, seed + 2),
        ('test', pools['test'], TRAIN_TEMPLATES, n_test, seed + 3),
        ('ood', pools['test'], OOD_TEMPLATES, n_ood, seed + 4),
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, name_pool, templates, n, this_seed in configs:
        rows = make_examples(name_pool, templates, n, this_seed)
        (out_dir / f'{label}.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
    (out_dir / 'metadata.json').write_text(json.dumps({
        'seed': seed, 'names': pools, 'templates_train': TRAIN_TEMPLATES, 'templates_ood': OOD_TEMPLATES,
        'condition': 'The recipient (IO) appears once. The subject appears twice.',
        'warning': 'Train, dev, and test use different names. OOD uses test names in new sentence patterns.'
    }, indent=2), encoding='utf8')
    return pools


def load_examples(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
