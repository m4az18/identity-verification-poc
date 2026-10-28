"""Download AT&T/ORL research faces; keep all images and manifests local."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def main():
    root = Path('data/olivetti')
    root.mkdir(parents=True, exist_ok=True)
    faces = fetch_olivetti_faces(data_home='data/cache', shuffle=False)
    conditions = ['clear', 'blurred', 'dark', 'bright', 'partial_obstruction']
    for index, face in enumerate(faces.images):
        image = np.uint8(np.clip(face * 255, 0, 255))
        for condition in conditions:
            modified = image.copy()
            if condition == 'blurred':
                modified = cv2.GaussianBlur(image, (7, 7), 2)
            elif condition == 'dark':
                modified = np.uint8(image * 0.45)
            elif condition == 'bright':
                modified = np.uint8(np.clip(image.astype(float) * 1.5, 0, 255))
            elif condition == 'partial_obstruction':
                modified[42:, :] = 127
            cv2.imwrite(str(root / f'{index}_{condition}.png'), modified)
    rows = []
    for split, subjects in [('calibration', list(range(20))), ('test', list(range(20, 40)))]:
        for subject in subjects:
            other = subjects[(subjects.index(subject) + 1) % len(subjects)]
            for j in range(1, 10):
                for condition in conditions:
                    for genuine, second in [(1, subject), (0, other)]:
                        rows.append(dict(pair_id=f'{split}-{subject}-{j}-{condition}-{genuine}',
                            split=split, label=genuine, subject_a=subject, subject_b=second,
                            image_a=f'{subject * 10}_clear.png',
                            image_b=f'{second * 10 + j}_{condition}.png', condition=condition))
    with (root / 'pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'provenance.json').write_text(json.dumps({
        'source': 'https://cam-orl.co.uk/facedatabase.html/',
        'credit': 'AT&T Laboratories Cambridge',
        'purpose': 'Local research evaluation; no redistribution of images',
        'sha256_images': hashlib.sha256(faces.images.tobytes()).hexdigest(),
        'limitations': ['64x64 grayscale', 'No verified demographic or skin tone labels',
            'No old ID/recent photo pairs', 'No real masks',
            'Blur, exposure and obstruction are synthetic stress tests',
            'Natural pose/glasses variation is not individually annotated']}, indent=2))
    print(f'Prepared {len(rows)} pairs from 40 identities in {root}')


if __name__ == '__main__':
    main()
