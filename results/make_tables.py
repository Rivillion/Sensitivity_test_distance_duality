"""Regenerate manuscript Tables I and III-X and the Sec. IV.A/IV.B quantities.

Table II (Jeffreys' scale for Delta AIC) is a convention and is not generated.
Posterior intervals come from the saved weighted draws; likelihood improvements
from the optimized minima in the run summaries and results/minima.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

from results import calibration_table, posterior_tables
from src.background import c_kms
from src.calibration import MB_CALIBRATION, RD_CALIBRATION
from src.minima import make_sn
from src.posterior_io import MODELS, SAMPLE_KEYS, load_run
from src.release_io import repo_relative, sha256, validate_completed, verify_inputs
from src.sampler import PRIOR_BOUNDS, compute_getdist_constraints

ROOT = Path(__file__).resolve().parent.parent
ROMAN = {1: 'I', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'}
LABELS = dict(zip(SAMPLE_KEYS, ('Pantheon+', 'DES-Dovekie', 'Union3', 'Union3.1')))
DE = r'$\Delta\chi^2_{\rm DE}$'
DE_EPS = r'$\Delta\chi^2_{{\rm DE}|\varepsilon}$'
F_RED = r'$f_{\rm red}$ (\%)'
# (key in results/minima/tracers.json, Table VI row label)
TRACERS = (('None (all 13)', 'None (full DR2)'), ('BGS', 'BGS'), ('LRG1', 'LRG1'),
           ('LRG2', 'LRG2'), ('LRG3+ELG1', 'LRG3+ELG1'), ('ELG2', 'ELG2'), ('QSO', 'QSO'),
           ('Lyalpha', r'Ly$\alpha$'))
# The first cutoff of each sample is the baseline analysis (Table VII caption).
CUTS = (('pantheon', ('0.010', '0.023', '0.050', '0.100')), ('dovekie', ('0.025', '0.050', '0.100')))


def n_sigma(delta_chi2, dof=2):
    """Gaussian-equivalent significance of a chi-square improvement, Eqs. (19)-(20)."""
    return float(norm.isf(chi2.sf(max(delta_chi2, 0.0), dof) / 2))


def reduction(delta, delta_given_extension):
    """Reduction fraction f_red of Eq. (17), in percent."""
    return 100 * (1 - delta_given_extension / delta)


def write_table(out, number, headers, rows, metadata, *, tex_header=None, rules=()):
    """Write a tabular fragment, a JSON record and return the Markdown section."""
    stem = f'table_{number:02d}'
    tex = [r'\begin{tabular}{l' + 'c' * (len(headers) - 1) + '}', r'\hline\hline',
           *(tex_header or [' & '.join(headers) + r' \\']), r'\hline']
    for index, row in enumerate(rows):
        if index in rules:
            tex.append(r'\hline')
        tex.append(' & '.join(map(str, row)) + r' \\')
    tex += [r'\hline\hline', r'\end{tabular}']
    (out / f'{stem}.tex').write_text('\n'.join(tex) + '\n')
    (out / f'{stem}.json').write_text(json.dumps({'columns': headers, 'rows': rows, **metadata},
                                                 indent=2, allow_nan=False) + '\n')
    return '\n'.join([f'## Table {ROMAN[number]}', '', '| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], ''])


def table_01(out):
    rows = []
    for sample in SAMPLE_KEYS:
        z = np.asarray(make_sn(sample).z_cmb)
        rows.append([LABELS[sample], len(z), f'{z.min():.3f}--{z.max():.3f}'])
    return write_table(out, 1, ['Sample', 'Entries/nodes', 'Redshift range'], rows, {
        'redshift': 'Hubble-diagram redshift zHD (node redshifts for Union)',
        'selection': 'Pantheon+: zHD >= 0.01 and no Cepheid calibrators; others as released'})


def tables_03_04_05(out, baseline, provenance, profiles, minima_sha256, non_sn):
    bao_only = non_sn['bao']['delta_de']
    rows3 = [['BAO only', f'{bao_only:.2f} / {n_sigma(bao_only):.2f}', '------', '------']]
    rows4, rows5 = [], []
    for sample, runs in baseline.items():
        chi = {model: run['minimum_chi2'] for model, run in runs.items()}
        d = chi['lcdm'] - chi['w0wa']
        e = chi['lcdm-epsilon'] - chi['w0wa-epsilon']
        rows3.append([LABELS[sample], f'{d:.2f} / {n_sigma(d):.2f}', f'{e:.2f} / {n_sigma(e):.2f}',
                      f'{reduction(d, e):.0f}'])
        non_nested = chi['lcdm-epsilon'] - chi['w0wa']
        # w0waCDM has one more free parameter than LambdaCDM + epsilon.
        rows4.append([LABELS[sample], f'{non_nested:.2f}', f'{non_nested - 2:.2f}'])
        c = profiles['controls'][sample]
        rows5.append([LABELS[sample]] + [f'{v:.2f}' for v in (
            d, e, c['linear']['delta_de'], c['saturating']['delta_de'], c['curvature']['delta_de'])])
    report = [write_table(out, 3, ['Sample', DE + r' / $n_\sigma$', DE_EPS + r' / $n_\sigma$', F_RED],
                          rows3, {'runs': provenance, 'bao_only_minima': 'results/minima/non-sn.json',
                                  'significance': 'Eqs. (19)-(20): two degrees of freedom, two-sided Gaussian equivalent'},
                          rules=(1,))]
    report.append(write_table(out, 4, ['Sample', r'$\Delta\chi^2$', r'$\Delta{\rm AIC}$'], rows4, {
        'runs': provenance,
        'definition': 'Delta chi2 = min chi2(LambdaCDM+epsilon) - min chi2(w0waCDM); Delta AIC = Delta chi2 - 2'}))
    report.append(write_table(out, 5, ['Sample', DE, DE_EPS, r'$\Delta\chi^2_{{\rm DE}|\kappa}$',
                                       r'$\Delta\chi^2_{{\rm DE}|\lambda}$', r'$\Delta\chi^2_{{\rm DE}|\Omega_k}$'],
                              rows5, {'runs': provenance, 'minima_sha256': minima_sha256['controls'],
                                      'controls': 'kappa: linear z; lambda: z/(1+z); Omega_k: free curvature with epsilon = 0'}))
    return report


def table_06(out, profiles, minima_sha256, tracer_epsilon, robustness_provenance):
    headers, rows = ['Dropped tracer'], []
    for sample in SAMPLE_KEYS:
        headers += [f'{LABELS[sample]} {DE}', f'{LABELS[sample]} {DE_EPS}', f'{LABELS[sample]} {F_RED}']
    for key, label in TRACERS:
        row = [label]
        for sample in SAMPLE_KEYS:
            r = profiles['tracers'][sample][key]
            row += [f"{r['delta_de']:.2f}", f"{r['delta_de_given_tilt']:.2f}",
                    f"{reduction(r['delta_de'], r['delta_de_given_tilt']):.0f}"]
        rows.append(row)
    tex_header = [' & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{LABELS[s]}}}' for s in SAMPLE_KEYS) + r' \\',
                  'Dropped tracer' + (' & ' + ' & '.join((DE, DE_EPS, r'$f_{\rm red}$(\%)'))) * 4 + r' \\']
    # The text states that epsilon stays stable under each deletion without tabulating it.
    stability = {
        'pantheon_lcdm_epsilon_posterior': tracer_epsilon,
        'best_fit_epsilon': {sample: {key: {model: profiles['tracers'][sample][key]['fits'][model]['tilt']
                                            for model in ('lcdm-epsilon', 'w0wa-epsilon')}
                                      for key, _ in TRACERS} for sample in SAMPLE_KEYS},
    }
    return write_table(out, 6, headers, rows, {'minima_sha256': minima_sha256['tracers'],
                                               'epsilon_stability': stability,
                                               'runs': robustness_provenance}, tex_header=tex_header)


def table_07(out, profiles, minima_sha256, baseline, cut_epsilon, robustness_provenance):
    rows, rules = [], []
    for sample, cutoffs in CUTS:
        rules.append(len(rows))
        for index, cutoff in enumerate(cutoffs):
            r = profiles['cuts'][sample][cutoff]
            q = baseline[sample]['lcdm-epsilon']['epsilon'] if index == 0 else cut_epsilon[(sample, cutoff)]
            # Symmetrized with the larger of the two 68% half-widths.
            width = max(q['median'] - q['q16'], q['q84'] - q['median'])
            rows.append([LABELS[sample] if index == 0 else '', cutoff, r['n_sn'],
                         f"${q['median']:.3f} \\pm {width:.3f}$",
                         f"{r['delta_de']:.2f}", f"{r['delta_de_given_tilt']:.2f}"])
    return write_table(out, 7, ['Sample', r'$z_{\min}$', r'$N_{\rm SN}$', r'$\varepsilon_{\Lambda{\rm CDM}}$', DE, DE_EPS],
                       rows, {'minima_sha256': minima_sha256['cuts'], 'runs': robustness_provenance,
                              'epsilon': 'LambdaCDM + epsilon posterior median; first row per sample from the baseline run',
                              'boundary_minima': 'w0waCDM minima for DES-Dovekie zmin = 0.100 lie on the Omega_m = 0.1 prior boundary'},
                       rules=tuple(rules[1:]))


def table_09(out):
    def uniform(name):
        low, high = PRIOR_BOUNDS[name]
        if np.isclose(low, c_kms / 100) and np.isclose(high, c_kms / 10):
            return r'$\mathcal{U}(c/100, c/10)$'
        return rf'$\mathcal{{U}}({low:g}, {high:g})$'
    rows = [[r'$\Omega_{\rm m}$', uniform('Omega_m'), 'All backgrounds'],
            [r'$w_0$', uniform('w0'), 'CPL'], [r'$w_a$', uniform('wa'), 'CPL'],
            [r'$\varepsilon$', uniform('epsilon'), 'Tilt free'],
            [r'$H_0r_{\rm d}$ (km\,s$^{-1}$)', uniform('H0rd'), 'BAO without CC'],
            [r'$\nu$ (mag)', uniform('nu'), 'SN intercept'],
            [r'$H_0$ (km\,s$^{-1}$\,Mpc$^{-1}$)', uniform('H0'), 'BAO with CC'],
            [r'$r_{\rm d}$ (Mpc)', uniform('r_d'), 'With CC'],
            [r'$r_{\rm d}$ (Mpc)', rf'${RD_CALIBRATION[0]} \pm {RD_CALIBRATION[1]}$', 'Calibration only'],
            [r'$M_{\rm B}$ (mag)', rf'${MB_CALIBRATION[0]} \pm {MB_CALIBRATION[1]}$', 'Calibration only']]
    return write_table(out, 9, ['Parameter', 'Prior', 'Application'], rows,
                       {'source': 'src/sampler.py PRIOR_BOUNDS and src/calibration.py'}, rules=(8,))


def posterior_table(out, number, root, include_cc):
    rows = posterior_tables.collect_rows(root, include_cc=include_cc)
    (out / f'table_{number:02d}.tex').write_text(posterior_tables.render_latex(rows, include_cc=include_cc))
    (out / f'table_{number:02d}.json').write_text(json.dumps({
        'rows': rows, 'include_cc': include_cc,
        'quantile_method': 'GetDist weighted equal-tail percentiles; ignore_rows=0'}, indent=2, allow_nan=False) + '\n')
    return rows


def section_4a(out, baseline_root, non_sn):
    """BAO, CC and BAO+CC constraints and the BAO+SN+CC dark-energy significances."""
    standalone = {}
    for slug in ('desi', 'cc', 'desi+cc'):
        standalone[slug] = {}
        for model in ('lcdm', 'w0wa'):
            directory = baseline_root / slug / model
            summary = validate_completed(directory)
            with np.load(directory / 'posterior.npz', allow_pickle=False) as p:
                q = compute_getdist_constraints(p['samples'], p['weights'], p['log_likelihoods'],
                                                p['parameter_names'].tolist())
            standalone[slug][model] = {'minimum_chi2': summary['minimum_chi2'], 'constraints': q,
                                       'posterior_sha256': sha256(directory / 'posterior.npz')}
    with_cc = {}
    for sample in SAMPLE_KEYS:
        chi = {model: load_run(baseline_root, sample, model, include_cc=True)['minimum_chi2'] for model in MODELS}
        d, e = chi['lcdm'] - chi['w0wa'], chi['lcdm-epsilon'] - chi['w0wa-epsilon']
        with_cc[sample] = {'delta_de': d, 'n_sigma': n_sigma(d), 'delta_de_given_tilt': e,
                           'n_sigma_given_tilt': n_sigma(e)}
    result = {'non_sn_delta_de': {name: non_sn[name]['delta_de'] for name in ('bao', 'cc', 'bao_cc')},
              'bao_sn_cc_significance': with_cc, 'posteriors': standalone}
    (out / 'section4a.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    q = standalone['desi+cc']['lcdm']['constraints']
    lines = ['## Section IV.A', '',
             'Dark-energy improvement without SN: ' + ', '.join(
                 f"{name} {value:.2f}" for name, value in result['non_sn_delta_de'].items()) + '.', '',
             f"BAO+CC LambdaCDM: H0 = {q['H0']['median']:.1f} +{q['H0']['plus']:.1f} -{q['H0']['minus']:.1f} km/s/Mpc, "
             f"r_d = {q['r_d']['median']:.1f} +{q['r_d']['plus']:.1f} -{q['r_d']['minus']:.1f} Mpc.", '',
             '| Sample (BAO+SN+CC) | ' + DE + r' / $n_\sigma$ | ' + DE_EPS + r' / $n_\sigma$ |', '| --- | --- | --- |']
    lines += [f"| {LABELS[s]} | {v['delta_de']:.2f} / {v['n_sigma']:.1f} | "
              f"{v['delta_de_given_tilt']:.2f} / {v['n_sigma_given_tilt']:.1f} |" for s, v in with_cc.items()]
    return '\n'.join(lines + [''])


def robustness_epsilon(root):
    """Weighted LambdaCDM + epsilon intervals for the tracer-deletion and cutoff runs."""
    tracer, cut, provenance = {}, {}, {}
    for job in json.loads((ROOT / 'results/jobs_robustness.json').read_text()):
        directory = root / job['case_id']
        summary = validate_completed(directory)
        selection = summary['resume_signature']['robustness_selection']
        if selection['case_id'] != job['case_id'] or selection['bao_drop_indices'] != job['bao_drop_indices']:
            raise ValueError(f'Incorrect robustness selection: {directory}')
        with np.load(directory / 'posterior.npz', allow_pickle=False) as p:
            c = compute_getdist_constraints(p['samples'], p['weights'], p['log_likelihoods'],
                                            p['parameter_names'].tolist())['epsilon']
        q = {'median': c['median'], 'q16': c['lower_68'], 'q84': c['upper_68']}
        provenance[job['case_id']] = {'posterior_sha256': sha256(directory / 'posterior.npz'),
                                      'summary_sha256': sha256(directory / 'summary.json')}
        if job['group'] == 'tracer':
            tracer['None (all 13)' if job['label'] == 'None' else job['label']] = q
        else:
            cut[(job['sample'], f"{job['zmin']:.3f}")] = q
    return tracer, cut, provenance


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
    names = ('controls', 'tracers', 'cuts')
    profiles = {name: json.loads((args.minima_dir / f'{name}.json').read_text()) for name in names}
    minima_sha256 = {name: sha256(args.minima_dir / f'{name}.json') for name in names}
    non_sn = json.loads((args.minima_dir / 'non-sn.json').read_text())
    baseline = {s: {m: load_run(args.baseline_root, s, m) for m in MODELS} for s in SAMPLE_KEYS}
    provenance = {s: {m: r['provenance'] for m, r in runs.items()} for s, runs in baseline.items()}
    tracer_epsilon, cut_epsilon, robustness_provenance = robustness_epsilon(args.robustness_root)

    report = ['# Reproduced manuscript tables', '',
              'Numbering follows the manuscript. Table II (Jeffreys scale) is a convention and is not generated. '
              'Posterior intervals use weighted GetDist equal-tail percentiles; likelihood improvements use optimized minima.', '']
    report.append(table_01(out))
    report += tables_03_04_05(out, baseline, provenance, profiles, minima_sha256, non_sn)
    report.append(table_06(out, profiles, minima_sha256, tracer_epsilon, robustness_provenance))
    report.append(table_07(out, profiles, minima_sha256, baseline, cut_epsilon, robustness_provenance))
    posterior_table(out, 8, args.baseline_root, include_cc=True)
    report += ['## Table VIII', '', 'BAO+SN+CC posterior table: `table_08.tex` (REVTeX `table*`).', '']
    report.append(table_09(out))
    posterior_table(out, 10, args.baseline_root, include_cc=False)
    report += ['## Table X', '', 'BAO+SN posterior table: `table_10.tex` (REVTeX `table*`).', '']
    report.append(section_4a(out, args.baseline_root, non_sn))
    calibrated = calibration_table.main(args.calibration_root, out)
    report += ['## Section IV.B', '', (out / 'section4b.md').read_text().split('\n', 2)[2]]
    (out / 'tables.md').write_text('\n'.join(report))
    print(f'Wrote Tables I, III-X and Sec. IV.A/IV.B quantities to {repo_relative(out)}')
    return calibrated


if __name__ == '__main__':
    main()
