"""Descriptive paired uncertainty conditional on the supplied model seeds."""
import numpy as np

VARIANTS = ('short', 'no_context', 'long32768')

def summarize(meta, sparse_runs, dense_runs, draws=20000, seed=2026091671):
    assert len(sparse_runs) == len(dense_runs) and len(sparse_runs) >= 1
    article_order = [x['article_hash'] for x in meta if x['variant'] == 'short']
    assert len(article_order) == len(set(article_order))
    n = len(article_order)
    def array(run):
        assert len(run) == len(meta)
        by = {t: [] for t in VARIANTS}
        for x, y in zip(run, meta):
            assert all(x[k] == y[k] for k in ('item_id', 'variant', 'gold'))
            assert x['correct'] == (x['prediction'] == x['gold'])
            by[x['variant']].append(float(x['correct']))
        for t in VARIANTS:
            assert [x['article_hash'] for x in meta if x['variant'] == t] == article_order
        return {t: np.array(by[t]) for t in VARIANTS}
    sparse = [array(x) for x in sparse_runs]
    dense = [array(x) for x in dense_runs]
    indices = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    def interval(x):
        x = np.asarray(x)
        return dict(mean_pp=float(100*x.mean()), ci95_pp=(100*np.quantile(x[indices].mean(axis=1), [.025, .975])).tolist())
    comparison = {t: interval(np.mean([a[t]-b[t] for a,b in zip(sparse,dense)],axis=0)) for t in VARIANTS}
    benefits = {}
    for name, runs in [('sparse',sparse),('dense',dense)]:
        benefits[name] = interval(np.mean([a['long32768']-a['no_context'] for a in runs],axis=0))
    relative_benefit = interval(np.mean([(a['long32768']-a['no_context'])-(b['long32768']-b['no_context']) for a,b in zip(sparse,dense)],axis=0))
    return dict(articles=n,seeds=len(sparse),draws=draws,bootstrap_seed=seed,sparse_minus_dense=comparison,long_minus_no_context=benefits,relative_context_benefit=relative_benefit,scope='Resample paired articles after averaging differences over fixed seeds. Development-set descriptive intervals, not an equivalence or noninferiority test; excludes uncertainty over new training seeds and datasets.')
