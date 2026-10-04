"""Regenerate manuscript Tables III-X and calibrated Section IV.B quantities."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

from results import calibration_table, posterior_tables
from src.posterior_io import MODELS, SAMPLE_KEYS, load_run
from src.release_io import sha256, validate_completed, verify_inputs
from src.sampler import compute_getdist_constraints

ROOT = Path(__file__).resolve().parent.parent
LABELS = dict(zip(SAMPLE_KEYS, ('Pantheon+', 'DES-Dovekie', 'Union3', 'Union3.1')))
BLOCKS = ('None (all 13)', 'BGS', 'LRG1', 'LRG2', 'LRG3+ELG1', 'ELG2', 'QSO', 'Lyalpha')


def latex_cell(value):
    text = str(value)
    if text.startswith('$') and text.endswith('$'):
        return text
    return text.replace('%', r'\%').replace('_', r'\_')


def write_table(out, number, headers, rows, metadata):
    stem = f'table_{number:02d}'
    tex = [r'\begin{tabular}{' + 'l' * len(headers) + '}',
           ' & '.join(map(latex_cell, headers)) + r' \\', r'\hline']
    tex += [' & '.join(map(latex_cell, row)) + r' \\' for row in rows]
    tex.append(r'\end{tabular}')
    (out / (stem + '.tex')).write_text('\n'.join(tex) + '\n')
    (out / (stem + '.json')).write_text(json.dumps({'columns': headers, 'rows': rows, **metadata}, indent=2, allow_nan=False) + '\n')
    return '\n'.join([f'## Table {number}', '', '| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], ''])


def epsilon(q):
    return f"${q['median']:.4f}^{{+{q['plus']:.4f}}}_{{-{q['minus']:.4f}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=ROOT / 'results/runs/baseline')
    parser.add_argument('--calibration-root', type=Path, default=ROOT / 'results/runs/calibration')
    parser.add_argument('--robustness-root', type=Path, default=ROOT / 'results/runs/robustness')
    parser.add_argument('--minima-dir', type=Path, default=ROOT / 'results/minima')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/tables')
    args = parser.parse_args()
    verify_inputs()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = ['# Reproduced manuscript tables', '',
              'Posterior intervals use weighted GetDist equal-tail percentiles. Likelihood improvements use optimized minima. Profile intervals and frozen manuscript values are not table inputs.', '']
    profiles = {name: json.loads((args.minima_dir / f'{name}.json').read_text()) for name in ('controls', 'tracers', 'cuts')}
    provenance = {name: sha256(args.minima_dir / f'{name}.json') for name in profiles}
    baseline, rows3, rows5 = {}, [], []
    for sample in SAMPLE_KEYS:
        runs = {m: load_run(args.baseline_root, sample, m) for m in MODELS}
        minima = {m: r['minimum_chi2'] for m, r in runs.items()}
        baseline[sample] = runs
        d = minima['lcdm'] - minima['w0wa']
        e = minima['lcdm-epsilon'] - minima['w0wa-epsilon']
        sigma = lambda delta: norm.isf(chi2.sf(max(delta, 0), 2) / 2)
        rows3.append([LABELS[sample], f'{d:.2f}', f'{sigma(d):.2f}', f'{e:.2f}', f'{sigma(e):.2f}', f'{100*(1-e/d):.0f}%'])
        n = profiles['tracers'][sample]['None (all 13)']['n_sn'] + 13
        diff = minima['w0wa'] - minima['lcdm-epsilon']
        rows5.append([LABELS[sample], f"{minima['w0wa']:.2f}", f"{minima['lcdm-epsilon']:.2f}", f'{diff:.2f}', f'{diff+2:.2f}', f'{diff+math.log(n):.2f}'])
    base_provenance = {s: {m: r['provenance'] for m, r in runs.items()} for s, runs in baseline.items()}
    report.append(write_table(out, 3, ['Sample', r'$\Delta\chi^2$', r'$\sigma$', r'$\Delta\chi^2_{\varepsilon}$', r'$\sigma_{\varepsilon}$', 'Reduction'], rows3,
                              {'runs': base_provenance, 'significance': 'Two-sided Gaussian equivalent of chi-square survival probability with two degrees of freedom'}))
    report.append(write_table(out, 5, ['Sample', r'$\chi^2_{w_0w_a}$', r'$\chi^2_{\Lambda+\varepsilon}$', r'$\Delta\chi^2$', r'$\Delta{\rm AIC}$', r'$\Delta{\rm BIC}$'], rows5,
                              {'runs': base_provenance, 'sign': 'CPL minus LambdaCDM+epsilon', 'parameter_count_difference': 1}))
    for include_cc, number in ((False, 4), (True, 10)):
        rows = posterior_tables.collect_rows(args.baseline_root, include_cc=include_cc)
        (out / f'table_{number:02d}.tex').write_text(posterior_tables.render_latex(rows, include_cc=include_cc))
        (out / f'table_{number:02d}.json').write_text(json.dumps({'rows': rows, 'include_cc': include_cc,
            'quantile_method': 'GetDist weighted equal-tail percentiles; ignore_rows=0'}, indent=2, allow_nan=False) + '\n')
    rows6 = []
    for s in SAMPLE_KEYS:
        r = baseline[s]
        c = profiles['controls'][s]
        d = r['lcdm']['minimum_chi2'] - r['w0wa']['minimum_chi2']
        e = r['lcdm-epsilon']['minimum_chi2'] - r['w0wa-epsilon']['minimum_chi2']
        rows6.append([LABELS[s]] + [f'{v:.2f}' for v in (d, e, c['linear']['delta_de'], c['saturating']['delta_de'], c['curvature']['delta_de'])])
    report.append(write_table(out, 6, ['Sample', 'No extension', 'Log tilt', 'Linear tilt', 'Saturating tilt', 'Curvature'], rows6, {'minima_sha256': provenance, 'runs': base_provenance}))
    jobs = json.loads((ROOT / 'results/jobs_robustness.json').read_text())
    tracer_q, cut_q, robustness_provenance = {}, {}, {}
    for job in jobs:
        directory = args.robustness_root / job['case_id']
        summary = validate_completed(directory)
        selection = summary['resume_signature']['robustness_selection']
        if selection['case_id'] != job['case_id'] or selection['bao_drop_indices'] != job['bao_drop_indices']:
            raise ValueError(f'Incorrect robustness selection: {directory}')
        with np.load(directory / 'posterior.npz', allow_pickle=False) as p:
            names = p['parameter_names'].tolist()
            q = compute_getdist_constraints(p['samples'], p['weights'], p['log_likelihoods'], names)['epsilon']
        robustness_provenance[job['case_id']] = {'posterior_sha256': sha256(directory / 'posterior.npz'), 'summary_sha256': sha256(directory / 'summary.json')}
        if job['table'] == 'VII':
            tracer_q['None (all 13)' if job['label'] == 'None' else job['label']] = q
        else:
            cut_q[(job['sample'], f"{job['zmin']:.3f}")] = q
    cut_q[('pantheon', '0.010')] = tracer_q['None (all 13)']
    rows7 = []
    for block in BLOCKS + ('LRG1 + LRG2', 'BGS + LRG1 + LRG2'):
        r = profiles['tracers']['pantheon'][block]
        d, e = r['delta_de'], r['delta_de_given_tilt']
        rows7.append([block, epsilon(tracer_q[block]), f'{d:.2f}', f'{e:.2f}', f'{100*(1-e/d):.0f}%'])
    report.append(write_table(out, 7, ['Dropped block', r'$\varepsilon$', r'$\Delta\chi^2$', r'$\Delta\chi^2_{\varepsilon}$', 'Reduction'], rows7, {'minima_sha256': provenance, 'runs': robustness_provenance}))
    rows8 = []
    for block in BLOCKS:
        reductions = []
        for s in SAMPLE_KEYS:
            r = profiles['tracers'][s][block]
            reductions.append(f"{100*(1-r['delta_de_given_tilt']/r['delta_de']):.0f}%")
        rows8.append([block] + reductions)
    report.append(write_table(out, 8, ['Dropped block', *LABELS.values()], rows8, {'minima_sha256': provenance}))
    rows9 = []
    for s, cuts in (('pantheon', ('0.010', '0.023', '0.050', '0.100')), ('dovekie', ('0.025', '0.050', '0.100'))):
        for cutoff in cuts:
            r = profiles['cuts'][s][cutoff]
            rows9.append([LABELS[s], cutoff, r['n_sn'], epsilon(cut_q[(s, cutoff)]), f"{r['delta_de']:.2f}", f"{r['delta_de_given_tilt']:.2f}"])
    report.append(write_table(out, 9, ['Sample', r'$z_{\min}$', 'N SN', r'$\varepsilon$', r'$\Delta\chi^2$', r'$\Delta\chi^2_{\varepsilon}$'], rows9, {'minima_sha256': provenance, 'runs': robustness_provenance}))
    calibrated = calibration_table.main(args.calibration_root, out)
    standalone = {}
    for slug in ('desi', 'cc', 'desi+cc'):
        standalone[slug] = {}
        for model in ('lcdm', 'w0wa'):
            directory = args.baseline_root / slug / model
            summary = validate_completed(directory)
            with np.load(directory / 'posterior.npz', allow_pickle=False) as p:
                q = compute_getdist_constraints(p['samples'], p['weights'], p['log_likelihoods'], p['parameter_names'].tolist())
            standalone[slug][model] = {'minimum_chi2': summary['minimum_chi2'], 'constraints': q,
                'posterior_sha256': sha256(directory / 'posterior.npz')}
    (out / 'standalone.json').write_text(json.dumps(standalone, indent=2, allow_nan=False) + '\n')
    (out / 'tables.md').write_text('\n'.join(report))
    print(f'Wrote Tables III-X, standalone constraints and IV.B quantities to {out}')


if __name__ == '__main__':
    main()
