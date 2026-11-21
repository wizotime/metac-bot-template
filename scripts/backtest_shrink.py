"""Backtest shrinkage on SanBe-Bot's resolved binary questions (read-only API calls).

Usage: METACULUS_TOKEN=... python scripts/backtest_shrink.py [--user-id 310257]

For each resolved binary question we use our forecast active at spot-scoring time and
Metaculus' spot_peer_score. Peer score = 100 * (ln p_us - mean ln p_others), so the
others' mean log score can be recovered and the peer score of any modified forecast
recomputed (ignoring the tiny effect of our own forecast on the field).
"""
import argparse, datetime, json, math, os, urllib.request

def fetch(user_id):
    h = {"Authorization": "Token " + os.environ["METACULUS_TOKEN"]}
    out, seen = [], set()
    for off in range(0, 5000, 100):
        url = (f"https://www.metaculus.com/api/posts/?forecaster_id={user_id}"
               f"&statuses=resolved&limit=100&offset={off}&with_cp=true")
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60))
        new = [p for p in d["results"] if p["id"] not in seen]
        if not new:
            break
        seen.update(p["id"] for p in new); out += new
        if not d.get("next"):
            break
    return out

def rows_from(posts):
    rows, cp_visible = [], 0
    for p in posts:
        q = p.get("question") or {}
        if any((v or {}).get("latest") for v in (q.get("aggregations") or {}).values()):
            cp_visible += 1
        if q.get("type") != "binary" or q.get("resolution") not in ("yes", "no"):
            continue
        mf = q.get("my_forecasts") or {}; sd = mf.get("score_data") or {}; hist = mf.get("history") or []
        if not hist or sd.get("spot_peer_score") is None:
            continue
        st = q.get("spot_scoring_time") or q.get("cp_reveal_time")
        t = datetime.datetime.fromisoformat(st.replace("Z", "+00:00")).timestamp() if st else 1e18
        act = [x for x in hist if x["start_time"] <= t and (x["end_time"] is None or x["end_time"] > t)]
        py = (act[-1] if act else hist[-1])["forecast_values"][1]
        y = 1 if q["resolution"] == "yes" else 0
        po = py if y else 1 - py
        rows.append(dict(py=py, y=y, m=math.log(po) - sd["spot_peer_score"] / 100))
    return rows, cp_visible

def logit(p): return math.log(p / (1 - p))
def sig(x): return 1 / (1 + math.exp(-x))

def evaluate(rows, f):
    B = L = P = 0.0
    for r in rows:
        p = f(r["py"]); po = p if r["y"] else 1 - p
        B += (p - r["y"]) ** 2; L += math.log(po); P += 100 * (math.log(po) - r["m"])
    n = len(rows); return B / n, L / n, P / n

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--user-id", default="310257"); a = ap.parse_args()
    rows, cpv = rows_from(fetch(a.user_id))
    print(f"resolved binary questions: {len(rows)}; questions with visible CP: {cpv}")
    print("raw            Brier %.4f  meanLn %.4f  spotPeer %.2f" % evaluate(rows, lambda p: p))
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        print("shrink50 w=%.1f Brier %.4f  meanLn %.4f  spotPeer %.2f" % ((w,) + evaluate(rows, lambda p, w=w: sig((1 - w) * logit(p)))))
