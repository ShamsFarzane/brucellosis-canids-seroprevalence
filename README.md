# brucellosis-canids-seroprevalence
Misclassification-aware estimation of Brucella seroprevalence in wild canids

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Publication-quality DPI
PLOT_DPI = 300

# Duplicate and label key figures
IMPORTANT_FIGS = {
    'forest_plot.png',
    'overall_true_prevalence.png',
    'bayes_ppc_basic.png',
    'bayes_calibration_scatter.png',
    'bayes_loo_khat.png',
    'bayes_pooled_posterior.png',
}

__ORIG_SAVEFIG = plt.savefig

def _savefig_pub(path, *args, **kwargs):
    from pathlib import Path as _P
    kwargs.setdefault('dpi', PLOT_DPI)
    kwargs.setdefault('bbox_inches', 'tight')
    out = _P(str(path))
    __ORIG_SAVEFIG(out, *args, **kwargs)
    if out.name in IMPORTANT_FIGS:
        try:
            fig = plt.gcf()
            if fig.axes:
                t = fig.axes[0].get_title()
                if not t.startswith('Important'):
                    fig.axes[0].set_title(f'Important — {t}')
        except Exception:
            pass
        __ORIG_SAVEFIG(out.with_name('IMPORTANT_' + out.name), *args, **kwargs)

plt.savefig = _savefig_pub

np.random.seed(42)

SERUM_METHODS = {
    'RBT','ELISA','CFT','SAT','BPAT','RIV','AGID','2-ME','IFA','FPA','ICT','CIE','CIEF'
}

ALIASES = {
    'CARD':'RBT','RB':'RBT','RBCT':'RBT','ROSE BENGAL':'RBT',
    'BBA':'BPAT','BAPAT':'BPAT',
    '2ME':'2-ME','2-ME ':'2-ME','ME-2':'2-ME',
    'CIEP':'CIE','CIEF':'CIE','IFAT':'IFA',
    'ICT ':'ICT','LFA':'ICT',
    'AGID ':'AGID','AGPT':'AGID','FPA ':'FPA'
}
EXCLUDE_METHODS = {'CULTURE','PCR','QPCR','RT-PCR','MOLECULAR','SEQUENCING'}


# Priors (mean, sd) for sensitivity/specificity by method (modify if you have better test meta-analyses)
PRIORS = {
    'RBT':   {'Se': (0.90, 0.05), 'Sp': (0.98, 0.01)},
    'ELISA': {'Se': (0.95, 0.03), 'Sp': (0.98, 0.01)},
    'SAT':   {'Se': (0.85, 0.07), 'Sp': (0.97, 0.02)},
    'CFT':   {'Se': (0.90, 0.05), 'Sp': (0.99, 0.005)},
    'BPAT':  {'Se': (0.90, 0.05), 'Sp': (0.98, 0.01)},
    'RIV':   {'Se': (0.92, 0.05), 'Sp': (0.98, 0.02)},
    'AGID':  {'Se': (0.80, 0.08), 'Sp': (0.99, 0.01)},
    '2-ME':  {'Se': (0.88, 0.06), 'Sp': (0.98, 0.01)},
    'IFA':   {'Se': (0.90, 0.05), 'Sp': (0.98, 0.02)},
    'FPA':   {'Se': (0.93, 0.05), 'Sp': (0.98, 0.01)},
    'ICT':   {'Se': (0.90, 0.06), 'Sp': (0.98, 0.02)},
    'CIE':   {'Se': (0.85, 0.08), 'Sp': (0.98, 0.02)}
}

# ---------- Helpers for reading/cleaning Excel ----------

# ---------- Year parsing & ranges ----------
CANDIDATE_YEAR_COLS = ['Sampling Year','Year','Year of Sampling','Publication Year','Study Year']

def _parse_year_span(x):
    import re, pandas as _pd, numpy as _np
    if _pd.isna(x): return (None,None)
    s = str(x).strip()
    if not s: return (None,None)
    s = s.replace("–","-").replace("—","-").replace("/", "-")
    s = re.sub(r"\s+to\s+", "-", s, flags=re.I)
    # single year?
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m and "-" not in s:
        y = int(m.group(0)); return (y,y)
    # range like 2000-02 or 1999-2003
    m = re.search(r"\b((19|20)\d{2})\s*-\s*((19|20)?\d{2})\b", s)
    if m:
        y1 = int(m.group(1)); tail = m.group(3)
        y2 = int(tail) if len(tail)==4 else int(str(y1)[:2] + tail)
        if y2 < y1: y2 = y1
        return (y1,y2)
    # fallback: pick two numbers
    nums = re.findall(r"(?:19|20)?\d{2}", s)
    if len(nums)>=2 and len(nums[0])==4:
        ya = int(nums[0]); yb = int(nums[1]) if len(nums[1])==4 else int(str(ya)[:2]+nums[1])
        return (min(ya,yb), max(ya,yb))
    return (None,None)

def _ensure_year_range_columns(df, year_col=None):
    # If explicit YearStart/YearEnd exist, honor them
    if 'YearStart' in df.columns or 'YearEnd' in df.columns or 'Year End' in df.columns:
        ys = df.get('YearStart', df.get('Year Start', None))
        ye = df.get('YearEnd', df.get('Year End', None))
        df['YearStart'] = pd.to_numeric(ys, errors='coerce')
        df['YearEnd']   = pd.to_numeric(ye, errors='coerce')
        df['YearStart'] = df['YearStart'].fillna(df['YearEnd'])
        df['YearEnd']   = df['YearEnd'].fillna(df['YearStart'])
        df['YearMid'] = np.floor((df['YearStart'] + df['YearEnd'])/2.0).astype('Int64')
        return df
    # If a single combined "YearStart/YearEnd" exists, parse it
    if 'YearStart/YearEnd' in df.columns:
        ys, ye = [], []
        for v in df['YearStart/YearEnd'].tolist():
            a, b = _parse_year_span(v); ys.append(np.nan if a is None else a); ye.append(np.nan if b is None else b)
        df['YearStart'] = pd.to_numeric(pd.Series(ys), errors='coerce')
        df['YearEnd']   = pd.to_numeric(pd.Series(ye), errors='coerce')
        df['YearStart'] = df['YearStart'].fillna(df['YearEnd'])
        df['YearEnd']   = df['YearEnd'].fillna(df['YearStart'])
        df['YearMid'] = np.floor((df['YearStart'] + df['YearEnd'])/2.0).astype('Int64')
        return df
    # Otherwise, choose preferred year column
    col = None
    pref = ['Sampling Year','Year','Year of Sampling','Publication Year','Study Year']
    if year_col and year_col in df.columns: col = year_col
    else:
        for c in pref:
            if c in df.columns: col = c; break
    if col is None:
        df['YearStart'] = np.nan; df['YearEnd'] = np.nan; df['YearMid'] = pd.NA
        return df
    ys, ye = [], []
    for v in df[col].tolist():
        a,b = _parse_year_span(v)
        ys.append(np.nan if a is None else a); ye.append(np.nan if b is None else b)
    df['YearStart'] = pd.to_numeric(pd.Series(ys), errors='coerce')
    df['YearEnd']   = pd.to_numeric(pd.Series(ye), errors='coerce')
    df['YearMid'] = np.floor((df['YearStart'] + df['YearEnd'])/2.0).astype('Int64')
    return df



def canonicalize_method(x: object) -> str | None:
    if pd.isna(x): return None
    s = str(x).strip().upper()
    s = s.replace("–","-").replace("—","-")
    s = ALIASES.get(s, s)
    if s in EXCLUDE_METHODS: return None
    if s in SERUM_METHODS: return s
    if "ROSE" in s and "BENGAL" in s: return "RBT"
    if "AGID" in s: return "AGID"
    return None


def beta_params(mean, sd):
    var = sd**2
    tmp = mean*(1-mean)/var - 1
    a = mean * tmp
    b = (1-mean) * tmp
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        a, b = 2, 2
    return a, b

def _detect_header_row(df, required_cols=('Diagnostic Category','Diagnostic Method','Total','Positive')):
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.strip()
        if all(any(rc.lower()==str(c).strip().lower() for c in row) for rc in required_cols):
            df2 = df.iloc[i+1:].copy()
            df2.columns = row
            df2.reset_index(drop=True, inplace=True)
            return df2
    df1 = df.copy()
    df1.columns = df1.iloc[0]
    df2 = df1.iloc[1:].reset_index(drop=True)
    return df2


def _standardize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for cand in df.columns:
        lc = str(cand).strip().lower()
        # Diagnostic group / method
        if lc in {'diagnostic category','category'}: rename_map[cand] = 'Diagnostic Category'
        if lc in {'diagnostic method','method','test','assay','diagnostic'}: rename_map[cand] = 'Diagnostic Method'
        # Counts
        if lc in {'n','total tested','sample size','total','tested','samples','n_tested'}: rename_map[cand] = 'Total'
        if lc in {'positive','positives','pos','positive n','n_pos','cases'}: rename_map[cand] = 'Positive'
        # Entities
        if lc in {'country','nation'}: rename_map[cand] = 'Country'
        if lc in {'continent','region'}: rename_map[cand] = 'Continent'
        if lc in {'species','host','animal'}: rename_map[cand] = 'Species'
        if lc in {'study id','study','reference'}: rename_map[cand] = 'Study ID'
        # Years: we keep original names; parsed downstream
        if lc in {'year','year of sampling','publication year','study year','sampling year','yearstart/yearend'}:
            rename_map[cand] = cand
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def read_excel_loose(xlsx_path, sheet="auto"):
    xl = pd.ExcelFile(xlsx_path)
    if sheet == "auto":
        sheet_to_use = xl.sheet_names[0]
    else:
        sheet_to_use = sheet if sheet in xl.sheet_names else {s.lower(): s for s in xl.sheet_names}.get(sheet.lower(), xl.sheet_names[0])
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_to_use, header=None)
    df = _detect_header_row(df_raw)
    df = _standardize_columns(df)
    return df, sheet_to_use


def clean_serology_table(xlsx_path, sheet="auto"):
    df, used_sheet = read_excel_loose(xlsx_path, sheet)
    if 'Diagnostic Category' not in df.columns or 'Diagnostic Method' not in df.columns:
        raise ValueError("Required columns not found after header detection. Got columns: %s" % list(df.columns))
    # Year columns
    df = _ensure_year_range_columns(df, year_col=None)
    # Serology only
    sero = df[df['Diagnostic Category'].astype(str).str.strip().str.lower()=='serology'].copy()
    # Canonicalize method
    sero['Method'] = sero['Diagnostic Method'].map(canonicalize_method)
    sero = sero[sero['Method'].notna()]
    sero = sero[~sero['Method'].isin(EXCLUDE_METHODS)]
    # Counts
    for col in ['Positive','Total']:
        sero[col] = pd.to_numeric(sero[col], errors='coerce')
    sero = sero.dropna(subset=['Total','Positive'])
    sero = sero[sero['Total']>0].copy()
    sero['obs_prev'] = sero['Positive']/sero['Total']
    return sero, used_sheet


# ---------- Rogan–Gladen simulation + summaries ----------

def adjust_prevalence(sero, draws=10000, priors=PRIORS):
    rows = []
    adj_rows = []
    for idx, r in sero.iterrows():
        method = str(r['Method'])
        if method not in priors:
            continue
        pos = int(r['Positive']); n = int(r['Total']); p_obs = pos/n
        a_se,b_se = beta_params(*priors[method]['Se'])
        a_sp,b_sp = beta_params(*priors[method]['Sp'])
        se = np.random.beta(a_se,b_se,size=draws)
        sp = np.random.beta(a_sp,b_sp,size=draws)
        tp = (p_obs + sp - 1)/(se + sp - 1)
        tp = np.clip(tp, 0, 1)
        sim_pos = np.random.binomial(n, se*tp + (1-sp)*(1-tp), size=draws)
        adj_rows.append({
            'Study ID': r.get('Study ID', ''),
            'Country': r.get('Country', ''),
            'Continent': r.get('Continent', ''),
            'Species': r.get('Species', ''),
            'Method': method,
            'n': n,
            'pos': pos,
            'obs_prev': p_obs,
            'tp_mean': float(np.mean(tp)),
            'tp_median': float(np.median(tp)),
            'tp_ci_lower': float(np.quantile(tp, 0.025)),
            'tp_ci_upper': float(np.quantile(tp, 0.975)),
            'pval_ppc': float((sim_pos >= pos).mean()),
        })
        rows.append((idx,n,p_obs,a_se,b_se,a_sp,b_sp))
    adj = pd.DataFrame(adj_rows).sort_values('tp_mean', ascending=False).reset_index(drop=True)
    return adj, rows

def pooled_draws(rows, sero, draws=3000):
    continents = sorted(sero['Continent'].dropna().unique().tolist()) if 'Continent' in sero.columns else []
    pooled = np.zeros(draws)
    by_cont = {c: np.zeros(draws) for c in continents}
    for d in range(draws):
        numer = denom = 0.0
        num_c = {c:0.0 for c in continents}
        den_c = {c:0.0 for c in continents}
        for (idx, n, p_obs, a_se, b_se, a_sp, b_sp) in rows:
            se = np.random.beta(a_se,b_se)
            sp = np.random.beta(a_sp,b_sp)
            tp = (p_obs + sp - 1)/(se + sp - 1)
            tp = float(np.clip(tp,0,1))
            numer += n*tp; denom += n
            c = sero.loc[idx,'Continent'] if 'Continent' in sero.columns else None
            if isinstance(c,str):
                num_c[c] += n*tp; den_c[c] += n
        pooled[d] = numer/denom if denom else np.nan
        for c in continents:
            by_cont[c][d] = num_c[c]/den_c[c] if den_c[c] else np.nan
    return pooled, by_cont

def validate_model(sero, priors=PRIORS, draws=4000):
    cover = []; briers = []
    for idx, r in sero.iterrows():
        m = str(r['Diagnostic Method'])
        if m not in priors: continue
        n = int(r['Total']); pos = int(r['Positive']); p_obs = pos/n
        a_se,b_se = beta_params(*priors[m]['Se'])
        a_sp,b_sp = beta_params(*priors[m]['Sp'])
        se = np.random.beta(a_se,b_se,size=draws)
        sp = np.random.beta(a_sp,b_sp,size=draws)
        tp = (p_obs + sp - 1)/(se + sp - 1)
        tp = np.clip(tp,0,1)
        pred_prob = se*tp + (1-sp)*(1-tp)
        pred_pos = np.random.binomial(n, pred_prob)
        lo, hi = np.quantile(pred_pos, [0.025,0.975])
        cover.append((pos>=lo) & (pos<=hi))
        briers.append((p_obs - np.mean(pred_prob))**2)
    return float(np.mean(cover)), float(np.mean(briers))

# ---------- Plotting ----------

def make_plots(adj_df, pooled, by_cont, save_dir: Path, prefix: str = ""):
    save_dir.mkdir(parents=True, exist_ok=True)
    def nm(x): return f"{prefix}_{x}" if prefix else x

    # 1) Forest plot
    fig1 = plt.figure(figsize=(8, max(6, 0.25*len(adj_df))))
    y = np.arange(len(adj_df))
    plt.errorbar(adj_df['tp_mean'], y,
                 xerr=[adj_df['tp_mean']-adj_df['tp_ci_lower'], adj_df['tp_ci_upper']-adj_df['tp_mean']],
                 fmt='o', capsize=3)
    plt.scatter(adj_df['obs_prev'], y, marker='x')
    left_labels = (
        adj_df.get('Study ID', pd.Series(['']*len(adj_df))).astype(str) + ' | '
        + adj_df.get('Species', pd.Series(['']*len(adj_df))).astype(str).str.slice(0,25) + ' | '
        + adj_df['Method'].astype(str) + ' | '
        + adj_df.get('YearMid', pd.Series([pd.NA]*len(adj_df))).astype('Int64').astype(str)
    )
    plt.yticks(y, left_labels)
    plt.xlabel('Prevalence'); plt.title('Observed (x) vs Adjusted (o) Prevalence by Study')
    plt.tight_layout()
    plt.savefig(save_dir/nm("forest_plot.png"))
    plt.close(fig1)

    # 2) Overall posterior (simulation)
    fig2 = plt.figure(figsize=(7,5))
    plt.hist(pooled, bins=30, density=True, alpha=0.8)
    plt.axvline(np.mean(pooled), linestyle='--')
    plt.xlabel('Overall true prevalence'); plt.ylabel('Density'); plt.title('Posterior of Overall True Prevalence (RG)')
    plt.tight_layout()
    plt.savefig(save_dir/nm("overall_true_prevalence.png"))
    plt.close(fig2)

    # 3) By-continent posteriors
    for c, draws in by_cont.items():
        fig = plt.figure(figsize=(7,5))
        d = np.asarray(draws); d = d[np.isfinite(d)]
        plt.hist(d, bins=30, density=True, alpha=0.8)
        plt.axvline(np.mean(d), linestyle='--')
        plt.xlabel(f'True prevalence in {c}'); plt.ylabel('Density'); plt.title(f'Posterior of True Prevalence - {c}')
        plt.tight_layout()
        plt.savefig(save_dir/nm(f"true_prev_{c.replace(' ','_')}.png"))
        plt.close(fig)

def plot_calibration_scatter(n, y, q_hat, out_path):
    """Simple calibration: predicted vs observed per study."""
    obs = y / np.maximum(n, 1)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(q_hat, obs, marker='o')
    lims = [0,1]
    plt.plot(lims, lims, linestyle='--')
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel('Predicted observed prevalence (q_hat)')
    plt.ylabel('Observed prevalence (y/n)')
    plt.title('Calibration by study')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# ---------- Bayesian fit (wrapped so we can refit for sensitivity) ----------

def fit_and_save(adj_df, pooled_RG, out_dir: Path, prefix: str, args):
    import pymc as pm
    import arviz as az

    def nm(x): return f"{prefix}_{x}" if prefix else x
    def _logit(p):
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p/(1-p))

    # Data
    J = len(adj_df)
    n = adj_df['n'].values.astype("int64")
    y = adj_df['pos'].values.astype("int64")

    meths = adj_df['Method'].astype('category')
    m_idx = meths.cat.codes.values
    M = meths.cat.categories.size

    species = adj_df.get('Species', pd.Series(['']*J)).astype('category')
    s_idx = species.cat.codes.values
    S = species.cat.categories.size

    country = adj_df.get('Country', pd.Series(['']*J)).astype('category')
    c_idx = country.cat.codes.values
    C = country.cat.categories.size

    continent = adj_df.get('Continent', pd.Series(['']*J)).fillna('Unknown').astype('category')
    ct_idx = continent.cat.codes.values
    CT = continent.cat.categories.size

    # Prior center for prevalence (logit) — near RG pooled
    pooled_center = float(np.nanmean(pooled_RG)) if np.isfinite(np.nanmean(pooled_RG)) else 0.2
    mu0 = _logit(pooled_center)
    tau_mu = 1.5

    # Beta params for se/sp by method from PRIORS
    se_alpha = np.array([beta_params(*PRIORS[m]['Se'])[0] for m in meths.cat.categories], dtype=float)
    se_beta  = np.array([beta_params(*PRIORS[m]['Se'])[1] for m in meths.cat.categories], dtype=float)
    sp_alpha = np.array([beta_params(*PRIORS[m]['Sp'])[0] for m in meths.cat.categories], dtype=float)
    sp_beta  = np.array([beta_params(*PRIORS[m]['Sp'])[1] for m in meths.cat.categories], dtype=float)

    with pm.Model() as model:
        # Sensitivity & specificity per method
        se = pm.Beta('se', alpha=se_alpha, beta=se_beta, shape=M)
        sp = pm.Beta('sp', alpha=sp_alpha, beta=sp_beta, shape=M)

        # Global prevalence (logit scale)
        mu = pm.Normal('mu', mu=mu0, sigma=tau_mu)

        # Random effects (species, country) — robust option
        if args.robust_priors:
            sigma_s = pm.HalfStudentT('sigma_s', nu=3, sigma=1.0)
            sigma_c = pm.HalfStudentT('sigma_c', nu=3, sigma=1.0)
        else:
            sigma_s = pm.HalfNormal('sigma_s', sigma=1.0)
            sigma_c = pm.HalfNormal('sigma_c', sigma=1.0)

        a_s_raw = pm.Normal('a_s_raw', mu=0.0, sigma=1.0, shape=S)
        a_c_raw = pm.Normal('a_c_raw', mu=0.0, sigma=1.0, shape=C)
        a_s = pm.Deterministic('a_s', sigma_s * a_s_raw)
        a_c = pm.Deterministic('a_c', sigma_c * a_c_raw)

        # Optional continent random effects
        if args.continent_re:
            if args.robust_priors:
                sigma_ct = pm.HalfStudentT('sigma_ct', nu=3, sigma=1.0)
            else:
                sigma_ct = pm.HalfNormal('sigma_ct', sigma=1.0)
            a_ct_raw = pm.Normal('a_ct_raw', mu=0.0, sigma=1.0, shape=CT)
            a_ct = pm.Deterministic('a_ct', sigma_ct * a_ct_raw)
            ct_term = a_ct[ct_idx]
        else:
            ct_term = 0.0

        # Optional study-level heavy-tailed random effect (absorbs idiosyncratic studies)
        if args.study_re:
            if args.robust_priors:
                sigma_st = pm.HalfStudentT('sigma_st', nu=3, sigma=1.0)
            else:
                sigma_st = pm.HalfNormal('sigma_st', sigma=1.0)
            eps_st = pm.StudentT('eps_st', nu=3, mu=0.0, sigma=1.0, shape=J)
            a_st = pm.Deterministic('a_st', sigma_st * eps_st)
        else:
            a_st = 0.0

        # Study-level true prevalence (logit)
        logit_p = mu + a_s[s_idx] + a_c[c_idx] + ct_term + a_st
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))

        # Misclassification: Pr(test +)
        q_raw = se[m_idx] * p + (1 - sp[m_idx]) * (1 - p)
        q = pm.Deterministic('q', q_raw)

        # Likelihood: Binomial or Beta–Binomial (Gamma prior for concentration to avoid mode at 0)
        if args.beta_binomial:
            shape = 10.0
            rate = shape / max(1e-6, args.q_conc)   # mean = shape/rate = q_conc
            q_kappa = pm.Gamma('q_kappa', alpha=shape, beta=rate)
            q_safe = pm.Deterministic('q_safe', pm.math.clip(q, 1e-6, 1-1e-6))
            alpha_q = pm.Deterministic('alpha_q', q_safe * q_kappa)
            beta_q  = pm.Deterministic('beta_q', (1 - q_safe) * q_kappa)
            y_obs = pm.BetaBinomial('y_obs', n=n, alpha=alpha_q, beta=beta_q, observed=y)
        else:
            y_obs = pm.Binomial('y_obs', n=n, p=pm.math.clip(q, 1e-6, 1-1e-6), observed=y)

        # Overall pooled true prevalence (n-weighted)
        p_pooled = pm.Deterministic('p_pooled', pm.math.sum(p * n) / pm.math.sum(n))

        # ---- Sampler controls ----
        cores = args.bayes_cores if args.bayes_cores is not None else args.bayes_chains
        cores = max(1, min(cores, os.cpu_count() or cores))

        idata_posterior = pm.sample(
            draws=args.bayes_draws,
            tune=args.bayes_tune,
            chains=args.bayes_chains,
            cores=cores,
            random_seed=args.bayes_seed,
            target_accept=args.bayes_target_accept,
            init='jitter+adapt_diag',
            progressbar=True,
            idata_kwargs={"log_likelihood": True},
        )

        idata_ppc = pm.sample_posterior_predictive(
            idata_posterior,
            var_names=['y_obs'],
            random_seed=args.bayes_seed,
            progressbar=True,
            return_inferencedata=True,
        )

    # Save PPC separately
    ppc_nc_path = out_dir / nm("bayes_idata_ppc.nc")
    try:
        idata_ppc.to_netcdf(ppc_nc_path)
        print(f"[PyMC/ArviZ] Saved PPC InferenceData to: {ppc_nc_path}")
    except Exception as e:
        print("[PyMC/ArviZ] Could not save PPC InferenceData NetCDF:", e)

    # Merge or manually attach PPC
    try:
        idata = az.merge(idata_posterior, idata_ppc)
    except Exception as e:
        print("[ArviZ] merge() failed, proceeding with manual attach:", e)
        idata = idata_posterior
        try:
            idata.posterior_predictive = idata_ppc.posterior_predictive
        except Exception as e2:
            print("[ArviZ] Manual PPC attach failed:", e2)

    # Save InferenceData
    nc_path = out_dir / nm("bayes_idata.nc")
    try:
        idata.to_netcdf(nc_path)
        print(f"[PyMC/ArviZ] Saved InferenceData to: {nc_path}")
    except Exception as e:
        print("[PyMC/ArviZ] Could not save InferenceData NetCDF:", e)

    # ---------- ArviZ summaries & diagnostics ----------
    try:
        core_vars = ["mu", "sigma_s", "sigma_c", "p_pooled"]
        if args.continent_re and "sigma_ct" in model.named_vars:
            core_vars.insert(3, "sigma_ct")
        if args.study_re and "sigma_st" in model.named_vars:
            core_vars.insert(3, "sigma_st")
        summ = az.summary(idata, var_names=core_vars, round_to=3)
        summ.to_csv(out_dir / nm("bayes_summary_core.csv"))
        print(f"[ArviZ] Saved summary to: {out_dir/nm('bayes_summary_core.csv')}")

        # LOO/WAIC + export Pareto-k
        try:
            loo = az.loo(idata, pointwise=True)
            waic = az.waic(idata)
            with open(out_dir / nm("bayes_ic.txt"), "w") as f:
                f.write(f"LOO:\n{loo}\n\nWAIC:\n{waic}\n")
            print(f"[ArviZ] Saved LOO/WAIC to: {out_dir/nm('bayes_ic.txt')}")
            # Pareto-k export if available
            try:
                pk = None
                if hasattr(loo, "pareto_k") and loo.pareto_k is not None:
                    pk = np.asarray(loo.pareto_k)
                elif hasattr(loo, "pareto_khat") and loo.pareto_khat is not None:
                    pk = np.asarray(loo.pareto_khat)
                if pk is not None and len(pk) == J:
                    tbl = adj_df.copy()
                    tbl["pareto_k"] = pk
                    tbl = tbl.sort_values("pareto_k", ascending=False)
                    tbl.to_csv(out_dir / nm("bayes_pareto_k_by_study.csv"), index=False)
                    print(f"[ArviZ] Saved Pareto-k table to: {out_dir/nm('bayes_pareto_k_by_study.csv')}")
            except Exception as e:
                print("[ArviZ] Could not export Pareto-k table:", e)
        except Exception as e:
            print("[ArviZ] Could not compute LOO/WAIC:", e)

        # Trace / Rank / Energy / Autocorr
        try:
            az.plot_trace(idata, var_names=core_vars)
            plt.savefig(out_dir / nm("bayes_trace_core.png"))
            plt.close(plt.gcf())
        except Exception as e:
            print("[ArviZ] Trace plot failed:", e)
        try:
            az.plot_rank(idata, var_names=core_vars)
            plt.savefig(out_dir / nm("bayes_rank_core.png"))
            plt.close(plt.gcf())
        except Exception as e:
            print("[ArviZ] Rank plot failed:", e)
        try:
            az.plot_energy(idata)
            plt.savefig(out_dir / nm("bayes_energy.png"))
            plt.close(plt.gcf())
        except Exception as e:
            print("[ArviZ] Energy plot failed:", e)
        try:
            az.plot_autocorr(idata, var_names=["mu","p_pooled"])
            plt.savefig(out_dir / nm("bayes_autocorr.png"))
            plt.close(plt.gcf())
        except Exception as e:
            print("[ArviZ] Autocorr plot failed:", e)

        # k-hat plot (if available in your ArviZ)
        try:
            if hasattr(az, "plot_khat"):
                az.plot_khat(idata)
                plt.savefig(out_dir / nm("bayes_loo_khat.png"))
                plt.close(plt.gcf())
        except Exception as e:
            print("[ArviZ] k-hat plot failed:", e)

        # PPC plot
        try:
            idata_for_ppc = idata if hasattr(idata, "posterior_predictive") and idata.posterior_predictive is not None else idata_ppc
            az.plot_ppc(idata_for_ppc, num_pp_samples=100)
            plt.savefig(out_dir / nm("bayes_ppc.png"))
            plt.close(plt.gcf())
        except Exception as e:
            print("[ArviZ] PPC plot failed:", e)

    except Exception as e:
        print("[ArviZ] Not available or plotting failed; will still save basic arrays:", e)

    # ---------- Save posterior arrays + pooled posterior histogram ----------
    try:
        p_pooled_draws = np.asarray(idata.posterior["p_pooled"].values).reshape(-1)
        np.save(out_dir / nm("bayes_p_pooled_draws.npy"), p_pooled_draws)

        p_draws = np.asarray(idata.posterior["p"].values)
        p_draws = p_draws.reshape(-1, p_draws.shape[-1])
        np.save(out_dir / nm("bayes_p_draws_by_study.npy"), p_draws)

        fig = plt.figure(figsize=(7,5))
        plt.hist(p_pooled_draws, bins=30, density=True, alpha=0.8)
        plt.axvline(np.mean(p_pooled_draws), linestyle='--')
        plt.xlabel('Bayesian overall true prevalence'); plt.ylabel('Density')
        plt.title('Posterior of Overall True Prevalence (Bayes)')
        plt.tight_layout()
        plt.savefig(out_dir / nm("bayes_pooled_posterior.png"))
        plt.close(fig)
    except Exception as e:
        print("[PyMC] Could not save posterior arrays/plot:", e)

    # ---------- Basic PPC figure (fallback; no ArviZ needed) ----------
    try:
        if hasattr(idata_ppc, "posterior_predictive"):
            y_ppc = np.asarray(idata_ppc.posterior_predictive["y_obs"].values)
        else:
            y_ppc = None
        if y_ppc is not None:
            y_ppc = y_ppc.reshape(-1, J)
            obs_prop = float(np.sum(y) / np.sum(n))
            ppc_prop = y_ppc.sum(axis=1) / float(np.sum(n))

            fig = plt.figure(figsize=(7,5))
            plt.hist(ppc_prop, bins=30, density=True, alpha=0.8)
            plt.axvline(obs_prop, linestyle='--')
            plt.xlabel('Predicted overall observed prevalence'); plt.ylabel('Density')
            plt.title('Posterior Predictive Check (overall observed prevalence)')
            plt.tight_layout()
            plt.savefig(out_dir / nm("bayes_ppc_basic.png"))
            plt.close(fig)

        # Calibration scatter (predicted q vs observed y/n)
        try:
            # Posterior mean of q per study:
            q_post = np.asarray(idata.posterior["q"].values)  # (chain, draw, J)
            q_hat = q_post.reshape(-1, q_post.shape[-1]).mean(axis=0)
            plot_calibration_scatter(n, y, q_hat, out_dir / nm("bayes_calibration_scatter.png"))
        except Exception as e:
            print("[Diag] Calibration scatter failed:", e)

    except Exception as e:
        print("[PyMC] Could not make basic PPC figure:", e)

    
   
    try:
        import arviz as az
        notes = out_dir / nm("manuscript_notes.txt")
        # pooled prevalence
        try:
            p_pooled_draws = np.asarray(idata.posterior["p_pooled"].values).reshape(-1)
            p_mean = float(np.mean(p_pooled_draws))
            p_q = np.quantile(p_pooled_draws, [0.025,0.5,0.975]).tolist()
        except Exception:
            p_mean, p_q = float('nan'), [float('nan')]*3
        # observed overall and PPC overall
        obs_overall = float(np.sum(y)/np.sum(n)) if np.sum(n)>0 else float('nan')
        try:
            if hasattr(idata, 'posterior_predictive') and idata.posterior_predictive is not None and 'y_obs' in idata.posterior_predictive:
                yppc = np.asarray(idata.posterior_predictive['y_obs'].values).reshape(-1, J)
                qhat = (yppc.sum(axis=1)/float(np.sum(n)))
                qhat_mean = float(np.mean(qhat))
                qhat_q = np.quantile(qhat, [0.025,0.5,0.975]).tolist()
            else:
                qhat_mean, qhat_q = float('nan'), [float('nan')]*3
        except Exception:
            qhat_mean, qhat_q = float('nan'), [float('nan')]*3
        # BFMI, Rhat, ESS
        try:
            bfmi = az.bfmi(idata); bfmi_vals = np.asarray(bfmi).ravel().tolist()
        except Exception:
            bfmi_vals = []
        try:
            summ = az.summary(idata, var_names=['mu','p_pooled','sigma_s','sigma_c','sigma_ct','sigma_st'], round_to=4)
            max_rhat = float(np.nanmax(summ.get('r_hat', np.nan)))
            min_ess = float(np.nanmin(summ.get('ess_bulk', np.nan)))
        except Exception:
            max_rhat, min_ess = float('nan'), float('nan')
        # Pareto-k summary
        k_summary = {}
        try:
            loo_pw = az.loo(idata, pointwise=True)
            pk = None
            if hasattr(loo_pw, 'pareto_k') and loo_pw.pareto_k is not None:
                pk = np.asarray(loo_pw.pareto_k)
            elif hasattr(loo_pw, 'pareto_khat') and loo_pw.pareto_khat is not None:
                pk = np.asarray(loo_pw.pareto_khat)
            if pk is not None:
                k_summary = {
                    'n': int(pk.size),
                    'n_gt_0_5': int((pk>0.5).sum()),
                    'n_gt_0_7': int((pk>0.7).sum()),
                    'n_gt_1_0': int((pk>1.0).sum()),
                    'max_k': float(np.max(pk)),
                }
        except Exception:
            pass
        with open(notes, 'w', encoding='utf-8') as f:
            f.write('# Manuscript notes\n')
            f.write(f'p_pooled_mean = {p_mean:.4f}; 95%CrI = {p_q}\n')
            f.write(f'observed_overall = {obs_overall:.4f}\n')
            f.write(f'ppc_overall_predicted_mean = {qhat_mean:.4f}; 95%CrI = {qhat_q}\n')
            f.write(f'BFMI_per_chain = {bfmi_vals}\n')
            f.write(f'max_Rhat = {max_rhat}; min_ESS_bulk = {min_ess}\n')
            f.write(f'ParetoK = {k_summary}\n')
            # Year info
            yr_ok = (~adj_df.get('YearMid', pd.Series([pd.NA]*len(adj_df))).isna()).sum()
            f.write(f'YearMid available for {yr_ok} / {len(adj_df)} studies. Sampling Year prioritized; YearStart/YearEnd honored.\n')
    except Exception as _e:
        print('[Notes] Could not write manuscript notes:', _e)
    
        print("PyMC model fitted. Artifacts saved in:", out_dir)
    return idata

# ---------- CLI / main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="/data/projects/p809_Rabies/General/Brucellosis.xlsx",
                    help="Path to Brucellosis Excel file")
    ap.add_argument("--sheet", default="auto",
                    help='Sheet name or "auto" to pick first available (default: auto)')
    ap.add_argument("--draws", type=int, default=10000, help="Simulation draws for RG adjustment")
    ap.add_argument("--outdir", default="brucellosis_outputs", help="Directory to write outputs")
    ap.add_argument("--prefix", default="", help="Filename prefix for outputs, e.g. brucellosis3")
    ap.add_argument("--skip-bayes", action="store_true", help="Skip PyMC hierarchical model")

    # MCMC controls
    ap.add_argument("--bayes-draws", type=int, default=1000, help="MCMC posterior draws per chain")
    ap.add_argument("--bayes-tune", type=int, default=1000, help="MCMC tuning steps per chain")
    ap.add_argument("--bayes-chains", type=int, default=4, help="Number of MCMC chains")
    ap.add_argument("--bayes-cores", type=int, default=None, help="Parallel worker processes (default: chains)")
    ap.add_argument("--bayes-target-accept", type=float, default=0.95, help="NUTS target_accept (0.9–0.995)")
    ap.add_argument("--bayes-seed", type=int, default=42, help="Random seed")

    # Robustness & overdispersion
    ap.add_argument("--robust-priors", action="store_true",
                    help="Half-Student-T (nu=3) for group scales")
    ap.add_argument("--beta-binomial", action="store_true",
                    help="Use Beta–Binomial overdispersion for counts")
    ap.add_argument("--q-conc", type=float, default=150.0,
                    help="Mean concentration (kappa) for Beta–Binomial; higher = closer to Binomial")
    ap.add_argument("--continent-re", action="store_true",
                    help="Add continent random effects")
    ap.add_argument("--study-re", action="store_true",
                    help="Add study-level heavy-tailed random effects")

    # Sensitivity / influence
    ap.add_argument("--sensitivity-trim-k", type=int, default=0,
                    help="Second-pass sensitivity refit excluding top-K studies by Pareto-k")

    args = ap.parse_args()

    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    def nm(x): return f"{args.prefix}_{x}" if args.prefix else x

    # Data + RG adjustments
    sero, used_sheet = clean_serology_table(args.xlsx, sheet=args.sheet)
    adj_df, rows = adjust_prevalence(sero, draws=args.draws, priors=PRIORS)
    pooled, by_cont = pooled_draws(rows, sero, draws=3000)
    cover, brier = validate_model(sero, priors=PRIORS, draws=4000)

    print(f"Using sheet: {used_sheet}")
    mean = float(np.nanmean(pooled)); lo, hi = [float(x) for x in np.nanquantile(pooled, [0.025,0.975])]
    print("Overall true prevalence (mean, 95% CI):", mean, [lo, hi])
    print("Coverage (should be ~0.95):", cover)
    print("Brier-like score (lower is better):", brier)

    # Save RG outputs
    sero.to_csv(out_dir/nm("serology_clean.csv"), index=False)
    adj_df.to_csv(out_dir/nm("adjusted_prevalence_per_study.csv"), index=False)
    np.save(out_dir/nm("pooled_draws.npy"), pooled)
    for c, d in by_cont.items():
        np.save(out_dir/nm(f"pooled_draws_{c.replace(' ','_')}.npy"), d)

    # RG plots
    make_plots(adj_df, pooled, by_cont, save_dir=out_dir, prefix=args.prefix)

    # ---- Bayesian model ----
    if args.skip_bayes:
        print("Skipping PyMC hierarchical model as requested (--skip-bayes).")
        return

    # First fit
    idata_full = fit_and_save(adj_df, pooled, out_dir, args.prefix, args)

    # Optional second-pass trimmed sensitivity refit (exclude top-K influential)
    if args.sensitivity_trim_k and (out_dir / nm("bayes_pareto_k_by_study.csv")).exists():
        pk_tbl = pd.read_csv(out_dir / nm("bayes_pareto_k_by_study.csv"))
        # Use the *current* ordering to select top-K; map back to original rows using index if present
        # If "pareto_k" file lacks row identifiers, we assume row order aligns with adj_df order saved earlier.
        topK_mask = pk_tbl.sort_values("pareto_k", ascending=False).index[:args.sensitivity_trim_k]
        adj_df_trim = adj_df.drop(index=topK_mask).reset_index(drop=True)
        trim_prefix = f"{args.prefix}_trimK{args.sensitivity_trim_k}" if args.prefix else f"trimK{args.sensitivity_trim_k}"
        print(f"[SENS] Excluding top-{args.sensitivity_trim_k} influential studies and refitting...")
        # Recompute RG pooled center for the trimmed set
        # (optional; small effect) — reuse original pooled center for stability
        fit_and_save(adj_df_trim, pooled, out_dir, trim_prefix, args)

if __name__ == "__main__":
    main()
