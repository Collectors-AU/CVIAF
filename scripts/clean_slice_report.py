#!/usr/bin/env python3
"""Report observed clean error and abstention by declared domain and kind.

Input is one complete compare JSON. One row is one model/asset, not one image;
correlated assets do not constitute independent Bernoulli trials. Confidence
bounds are one-sided exact binomial bounds only under independent assets.
"""
import argparse
from collections import defaultdict
import json
from scipy.stats import beta


def upper(k, n, confidence=.95):
    if not n:
        return None
    return 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))


def report(data):
    rows = data['data_axis']['per_model']
    groups = defaultdict(list)
    for row in rows:
        if row['contributes_poison']:
            continue
        scene = row.get('declared_scene') or {}
        domain = {k: scene.get(k, 'unrecorded') for k in
                  ('terrain', 'season', 'illumination', 'sensor_noise', 'sensor_blur')}
        groups[(row['kind'], json.dumps(domain, sort_keys=True),
                str(row.get('calibration_stratum_match', 'unrecorded')))].append(row)
    output = []
    for (kind, domain, stratum), subset in sorted(groups.items()):
        n = len(subset)
        flags = sum(bool(r['cviaf']['flagged']) for r in subset)
        abstained = sum(bool(r['cviaf']['abstained']) for r in subset)
        output.append({'kind': kind, 'declared_domain': json.loads(domain),
                       'declared_stratum_match': stratum,
                       'n_clean_assets': n, 'n_flagged': flags,
                       'empirical_fpr': flags / n,
                       'fpr_upper_95_if_independent': upper(flags, n),
                       'n_abstained': abstained, 'abstention_rate': abstained / n,
                       'abstention_upper_95_if_independent': upper(abstained, n)})
    return {'slices': output, 'unit': 'asset',
            'warning': 'Bounds require independent clean assets; no cross-slice pooling, and a zero count is not proof of zero population risk.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison_json')
    args = parser.parse_args()
    with open(args.comparison_json) as fh:
        print(json.dumps(report(json.load(fh)), indent=2))


if __name__ == '__main__':
    main()
