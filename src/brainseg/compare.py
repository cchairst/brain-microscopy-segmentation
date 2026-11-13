"""Collect benchmark reports into one table of per-brain means with bootstrap confidence intervals."""

import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = ['run', 'method', 'sample_id', 'class', 'metric', 'n', 'mean', 'ci_low', 'ci_high']


def load_report(path):
    """Read a report.json, given the file or the run directory that holds it. Returns (run name, report)."""
    path = Path(path)
    file = path / 'report.json' if path.is_dir() else path
    return file.parent.name, json.loads(file.read_text())


def section_values(report):
    """Yield (sample_id, class, metric, value) for every annotated section; undefined values are skipped."""
    for section in report['sections']:
        m = section.get('metrics')
        if not m:
            continue
        values = [('macro_tissue', 'dice', m['macro_tissue']['dice'])]
        for name, rec in m['per_class'].items():
            values += [(name, 'dice', rec['dice']), (name, 'iou', rec['iou'])]
        for unit in ('hd95_px', 'hd95_um'):
            values += [(name, unit, v) for name, v in m.get(unit, {}).items()]
        ref, pred = m['volume_fraction']['reference'], m['volume_fraction']['prediction']
        values += [
            (name, 'vf_error', abs(pred[name] - ref[name])) for name in ref if None not in (ref[name], pred[name])
        ]
        for name, metric, v in values:
            if v is not None:
                yield section['sample_id'], name, metric, v


def bootstrap_ci(values, n_boot=2000, seed=0):
    """Mean and 95% percentile-bootstrap interval over sections; no interval for fewer than 2 sections."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()), None, None
    means = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(low), float(high)


def compare(paths, output, n_boot=2000, seed=0):
    """Summarize each report per brain, class and metric over its sections, write a CSV, and return the rows."""
    rows, runs = [], set()
    for path in paths:
        run, report = load_report(path)
        if run in runs:
            raise ValueError(f'Two reports share the run name {run!r}; give each run its own output directory.')
        runs.add(run)
        groups = {}
        for sample, name, metric, v in section_values(report):
            groups.setdefault((sample, name, metric), []).append(v)
        for (sample, name, metric), values in groups.items():
            mean, low, high = bootstrap_ci(values, n_boot, seed)
            rows.append(dict(zip(COLUMNS, [run, report['method'], sample, name, metric, len(values), mean, low, high])))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
