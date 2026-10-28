"""Evaluate local pairs using the application's exact inference pipeline."""
import argparse
import csv
import json
import math
import hashlib
import importlib.metadata
from collections import Counter
from pathlib import Path

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def save_csv(path, rows, fields=None):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, threshold):
    genuine = [r for r in rows if r['label'] == 1]
    impostor = [r for r in rows if r['label'] == 0]
    if not genuine or not impostor:
        raise ValueError('Both genuine and impostor pairs are required')
    fa = sum(r['score'] is not None and r['score'] >= threshold for r in impostor)
    ta = sum(r['score'] is not None and r['score'] >= threshold for r in genuine)
    return dict(threshold=threshold, FAR=fa / len(impostor),
                FRR=1 - ta / len(genuine), TAR=ta / len(genuine),
                TPR=ta / len(genuine), false_accepts=fa,
                false_rejects=len(genuine) - ta, genuine=len(genuine), impostor=len(impostor))


def roc_eer(rows):
    # Conditional on successful acquisition; failures are reported separately.
    rows = [r for r in rows if r['score'] is not None]
    scores = sorted({r['score'] for r in rows}, reverse=True)
    if not scores or {r['label'] for r in rows} != {0, 1}:
        return [], None
    curve = [metrics(rows, math.nextafter(scores[0], math.inf))]
    curve += [metrics(rows, score) for score in scores]
    for left, right in zip(curve, curve[1:]):
        a, b = left['FAR'] - left['FRR'], right['FAR'] - right['FRR']
        if a <= 0 <= b:
            weight = -a / (b - a) if b != a else 0
            return curve, left['FAR'] + weight * (right['FAR'] - left['FAR'])
    return curve, None


def validate(rows, root):
    seen, subjects = set(), {'calibration': set(), 'test': set()}
    paths = {'calibration': set(), 'test': set()}
    for r in rows:
        if r['pair_id'] in seen:
            raise ValueError('Duplicate pair ID')
        seen.add(r['pair_id'])
        r['label'] = int(r['label'])
        if r['label'] not in (0, 1) or r['split'] not in subjects:
            raise ValueError('Invalid label or split')
        if (r['subject_a'] == r['subject_b']) != bool(r['label']):
            raise ValueError('Identity labels disagree with pair label')
        subjects[r['split']].update([r['subject_a'], r['subject_b']])
        for key in ['image_a', 'image_b']:
            path = (root / r[key]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f'Missing image or path outside dataset: {r[key]}')
            paths[r['split']].add(path)
    if subjects['calibration'] & subjects['test'] or paths['calibration'] & paths['test']:
        raise ValueError('Calibration/test leakage')
    for split in subjects:
        if {r['label'] for r in rows if r['split'] == split} != {0, 1}:
            raise ValueError(f'{split} needs both pair classes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('evaluation-output'))
    parser.add_argument('--target-far', type=float, default=0.01)
    args = parser.parse_args()
    if not 0 <= args.target_far <= 1:
        parser.error('--target-far must be between zero and one')
    with args.manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    validate(rows, args.manifest.parent)
    import cv2
    import numpy as np
    from verification import analyze_image, load_models, MODEL_REVISIONS
    load_models()
    cache = {}
    def embedding(name):
        if name not in cache:
            try:
                image = cv2.imread(str(args.manifest.parent / name))
                if image is None:
                    raise ValueError('Unreadable image')
                _, face = analyze_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                cache[name] = (face.normed_embedding, None)
            except Exception as error:
                cache[name] = (None, f'{type(error).__name__}: {error}')
        return cache[name]
    for index, r in enumerate(rows):
        a, error_a = embedding(r['image_a'])
        b, error_b = embedding(r['image_b'])
        r['error'] = error_a or error_b
        r['score'] = None if r['error'] else float(np.clip(np.dot(a, b), -1, 1))
        for threshold in THRESHOLDS:
            r[f'prediction_{threshold:.2f}'] = int(r['score'] is not None and r['score'] >= threshold)
        if index % 200 == 0:
            print(f'Scored {index}/{len(rows)} pairs', flush=True)
    report = {'target_far': args.target_far, 'splits': {},
        'model_revisions': MODEL_REVISIONS,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'dependencies': {name: importlib.metadata.version(name) for name in
            ['ultralytics', 'insightface', 'onnxruntime', 'numpy', 'scikit-learn']}}
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ['calibration', 'test']:
        subset = [r for r in rows if r['split'] == split]
        curve, eer = roc_eer(subset)
        report['splits'][split] = dict(pairs=len(subset),
            failed_pairs=sum(r['score'] is None for r in subset),
            errors=dict(Counter(r['error'] for r in subset if r['error'])),
            thresholds=[metrics(subset, t) for t in THRESHOLDS],
            conditional_eer=eer,
            conditions={c: [metrics([r for r in subset if r['condition'] == c], t)
                for t in THRESHOLDS] for c in sorted({r['condition'] for r in subset})})
        save_csv(args.output / f'roc-{split}.csv', curve,
                 list(curve[0]) if curve else ['threshold', 'FAR', 'TPR'])
        if curve:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.plot([r['FAR'] for r in curve], [r['TPR'] for r in curve])
            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel('False acceptance rate')
            plt.ylabel('True acceptance rate')
            plt.title(f'{split}: ROC conditional on successful acquisition')
            plt.savefig(args.output / f'roc-{split}.png')
            plt.close()
    candidates = [m for m in report['splits']['calibration']['thresholds'] if m['FAR'] <= args.target_far]
    usable = {r['label'] for r in rows if r['split'] == 'calibration' and r['score'] is not None} == {0, 1}
    best = min(candidates, key=lambda m: (m['FRR'], -m['threshold'])) if candidates and usable else None
    report['recommended_research_threshold'] = best['threshold'] if best else None
    report['selection_policy'] = 'Minimize calibration FRR subject to empirical FAR target; highest threshold breaks ties. Test identities never select threshold.'
    report['failure_policy'] = 'Failed comparisons reject; threshold metrics include failures. ROC/EER exclude them.'
    report['limitations'] = 'Small correlated research sample; empirical FAR is not a population guarantee. Not calibrated for identity documents or demographic fairness.'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save_csv(args.output / 'scores.csv', rows)
    print(json.dumps({k: v for k, v in report.items() if k != 'splits'}, indent=2))


if __name__ == '__main__':
    main()
