"""Offline backtest: tail-mass floor at open bounds, applied to our submitted CDFs.
Delta ln(p) at the resolution ~ change in spot peer score / 100 (field held fixed)."""
import json, math, datetime, sys
import os
# resolved.json = output of scripts/backtest_shrink.py fetch() (posts with my_forecasts)
P = json.load(open(os.environ.get('RESOLVED_JSON', 'resolved.json')))

def ts(s): return datetime.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def used_cdf(q):
    h = (q.get('my_forecasts') or {}).get('history') or []
    if not h: return None
    st = q.get('spot_scoring_time') or q.get('cp_reveal_time'); t = ts(st) if st else 1e18
    act = [x for x in h if x['start_time'] <= t and (x['end_time'] is None or x['end_time'] > t)]
    return (act[-1] if act else h[-1])['forecast_values']

def inv(cdf, level):
    n = len(cdf)
    for i in range(1, n):
        if cdf[i] >= level:
            a, b = cdf[i-1], cdf[i]
            return ((i-1) + (level-a)/(b-a if b > a else 1)) / (n-1)
    return 1.0 + (level - cdf[-1])  # beyond upper bound

def remap(cdf, qL2, qU2):
    qL, qU = cdf[0], cdf[-1]
    xs = [0, qL, qU, 1]; ys = [0, qL2, qU2, 1]
    def g(v):
        for j in range(1, 4):
            if v <= xs[j] + 1e-15:
                if xs[j] == xs[j-1]: return ys[j]
                return ys[j-1] + (ys[j]-ys[j-1]) * (v-xs[j-1]) / (xs[j]-xs[j-1])
        return 1
    return [g(v) for v in cdf]

def res_prob(q, cdf):
    r = q['resolution']; s = q['scaling']
    if r == 'above_upper_bound': return 1 - cdf[-1]
    if r == 'below_lower_bound': return cdf[0]
    x = float(r); loc = (x - s['range_min']) / (s['range_max'] - s['range_min'])
    n = len(cdf); i = min(n-1, max(1, math.ceil(loc*(n-1) - 1e-9)))
    return cdf[i] - cdf[i-1]

def run(near, m, verbose=False):
    tot = {}; trig = 0; rows = []
    for p in P:
        q = p.get('question')
        if not q or q['type'] not in ('numeric', 'discrete') or q.get('resolution') in (None, 'annulled', 'ambiguous'): continue
        cdf = used_cdf(q)
        if not cdf: continue
        proj = 'FE' if any(t['id'] == 33121 for t in (p['projects'].get('tournament') or [])) else 'MB'
        qL2, qU2 = cdf[0], cdf[-1]; fired = []
        if q['open_upper_bound'] and inv(cdf, 0.9) >= 1 - near and 1 - cdf[-1] < m:
            qU2 = 1 - m; fired.append('U')
        if q['open_lower_bound'] and inv(cdf, 0.1) <= near and cdf[0] < m:
            qL2 = m; fired.append('L')
        new = remap(cdf, qL2, qU2) if fired else cdf
        d = math.log(res_prob(q, new)) - math.log(res_prob(q, cdf))
        tot.setdefault(proj, [0, 0, 0]); tot[proj][0] += 100*d; tot[proj][1] += 1; tot[proj][2] += bool(fired)
        if fired: rows.append((proj, p['id'], ''.join(fired), q['resolution'] if 'bound' in str(q['resolution']) else 'in', round(100*d, 1), p['title'][:60]))
    return tot, rows

if __name__ == '__main__':
    for near in (0.05, 0.10, 0.15, 0.20):
        for m in (0.10, 0.12, 0.13, 0.15):
            tot, rows = run(near, m)
            s = sum(v[0] for v in tot.values())
            print(f"near={near:.2f} floor={m:.2f}  " + "  ".join(f"{k}: n={v[1]} trig={v[2]} d={v[0]:+.1f}" for k, v in sorted(tot.items())) + f"  total={s:+.1f}")
    tot, rows = run(0.10, 0.13)
    for r in rows: print(r)
