"""
satellite_eval.py -- can satellite chlorophyll drive the 7-day bloom model?
--------------------------------------------------------------------------
Milestone 1 of the "going global" plan. At every station where we have sonde
truth, uses ONLY satellite chlorophyll (+ MUR SST) as model input and scores
against the SONDE label (sonde daily chl > own p75 within 7 d, onset rows =
sonde chl <= p75 today).

Pre-registered outputs, per product (viirs4k, dineof2k, olci4k, olci750, olci300):
  coverage       valid satellite days / sonde days, raw and after 3-day ffill
  observability  of sonde bloom onsets, share with >=1 and >=3 satellite
                 observations in the preceding 7 days (ceiling on recall)
  agreement      Spearman(sat, sonde chl) on co-valid days; kappa of
                 same-day p75 exceedance
  skill          onset rows, two feature representations
                 (obs = valid days only, lags = previous observations;
                  ffill3 = daily grid, 3-day forward fill):
                 zeroshot   frozen Narragansett model, chl quantile-rescaled
                 refit      GB rolling-origin CV on satellite features
                 chl_rule   satellite chl > c (c on val year)
                 climatology station-DOY rate
                 sonde_refit GB on the sonde's own features, same station-dates
                 refit_satlabel  refit scored on the satellite's OWN p75 label
Verdict per product (fixed): GO if refit lift >= 1.3 with CI_lo > 1.0 AND
refit lift > chl_rule lift AND observability(>=1 obs) >= 0.60.

Run from fork root, BASE env:  python -m src.transfer.satellite_eval
"""
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from src.transfer.transfer_eval import (TIER_A, add_label, boot_ci, build_daily,
                                        climatology_baseline, metrics, pick_t,
                                        quantile_map, rolling_refit, rule_baseline)

STATIONS = "data/transfer/stations_latlon.csv"
CACHE = "data/transfer/satellite"
PRODUCTS = ["olci300", "olci750", "dineof2k", "viirs4k", "olci4k"]
SOURCES = ["narragansett", "chesapeake", "nerrs", "cefas_full", "imos", "lake_erie", "sfbay"]
HORIZON = 7


def sonde_table(source):
    if source == "narragansett":
        d = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
        d["thr_p75"] = d.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
    else:
        p = f"data/transfer/{source}_daily.csv"
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p, parse_dates=["date"])
    d = add_label(d, "thr_p75").rename(columns={"bloom_fwd": "y_sonde", "thr_p75": "thr_sonde"})
    d["year"] = d.date.dt.year
    return d


def sat_series(product, source, station):
    p = f"{CACHE}/{product}/{source}__{station}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=["date"]).dropna(subset=["value"])
    return d if len(d) else None


def sonde_onsets(g, thr):
    """dates where sonde chl crosses above thr after >=3 consecutive days below."""
    below = (g.chl <= thr).astype(int).values; dates = g.date.values; out = []
    run = 0
    for i in range(len(g)):
        if below[i]:
            run += 1
        else:
            if run >= 3:
                out.append(dates[i])
            run = 0
    return pd.to_datetime(out)


def main():
    stations = pd.read_csv(STATIONS).dropna(subset=["lat", "lon"])
    pack = joblib.load("release/narragansett_bloom_model.joblib")
    nar_chl = pd.read_csv("data/narragansett_daily_features.csv").chl.values
    sonde = {s: sonde_table(s) for s in SOURCES}
    sonde = {k: v for k, v in sonde.items() if v is not None}

    cov, obs, agr, skill = [], [], [], []
    for product in PRODUCTS:
        frames = {"obs": [], "ffill3": []}
        for _, st in stations.iterrows():
            if st.source not in sonde:
                continue
            sd = sonde[st.source]; sd = sd[sd.station == st.station]
            if sd.empty:
                continue
            sat = sat_series(product, st.source, st.station)
            sst = sat_series("sst", st.source, st.station)
            if sat is None:
                continue
            lo, hi = sat.date.min(), sat.date.max()
            sd = sd[(sd.date >= lo) & (sd.date <= hi)]
            if len(sd) < 60:
                continue
            grid = pd.DataFrame({"date": pd.date_range(sd.date.min(), sd.date.max())})
            grid = grid.merge(sat[["date", "value"]], on="date", how="left")
            valid = grid.value.notna(); ff = grid.value.ffill(limit=3).notna()
            cov.append(dict(source=st.source, station=st.station, product=product, days=len(grid),
                            valid=int(valid.sum()), frac=valid.mean(), frac_ffill3=ff.mean()))
            thr = sd.thr_sonde.iloc[0]
            ons = sonde_onsets(sd.sort_values("date"), thr)
            if len(ons):
                vd = set(sat.date); n1 = n3 = 0
                for d0 in ons:
                    k = sum((d0 - pd.Timedelta(days=j)) in vd for j in range(1, HORIZON + 1))
                    n1 += k >= 1; n3 += k >= 3
                obs.append(dict(product=product, source=st.source, station=st.station,
                                onsets=len(ons), obs1=n1 / len(ons), obs3=n3 / len(ons)))
            m = sd[["date", "chl"]].merge(sat[["date", "value"]], on="date")
            if len(m) >= 30:
                agr.append(dict(source=st.source, station=st.station, product=product, n=len(m),
                                spearman=spearmanr(m.chl, m.value).correlation,
                                kappa=cohen_kappa_score(m.chl > thr, m.value > m.value.quantile(0.75))))
            for rep in ("obs", "ffill3"):
                s = grid.copy() if rep == "ffill3" else sat[["date", "value"]].copy()
                if rep == "ffill3":
                    s["value"] = s.value.ffill(limit=3)
                s = s.dropna(subset=["value"])
                if sst is not None:
                    s = s.merge(sst[["date", "value"]].rename(columns={"value": "temp_c"}), on="date", how="left")
                else:
                    s["temp_c"] = np.nan
                f = pd.DataFrame({"station": f"{st.source}_{st.station}", "datetime": s.date + pd.Timedelta(hours=12),
                                  "chl_ugl": s.value, "temp_c": s.temp_c, "salinity_psu": np.nan, "do_mgl": np.nan})
                frames[rep].append((f, sd, st))

        for rep, items in frames.items():
            if not items:
                continue
            parts = []
            for f, sd, st in items:
                day = build_daily(f, min_readings=1)
                day["thr_sat"] = day.chl.quantile(0.75)
                j = sd[["date", "chl", "thr_sonde", "y_sonde"]].rename(columns={"chl": "chl_sonde"})
                parts.append(day.merge(j, on="date", how="inner").assign(source=st.source))
            D = pd.concat(parts, ignore_index=True)
            D = add_label(D.assign(thr=D.thr_sat), "thr").rename(columns={"bloom_fwd": "y_sat"})
            D["bloom_fwd"] = D.y_sonde
            D = D.dropna(subset=["bloom_fwd"]).copy()
            D["thr"] = D.thr_sonde
            onset = (D.chl_sonde <= D.thr_sonde).values
            print(f"\n[{product} / {rep}] stations={D.station.nunique()} labelled={len(D):,} "
                  f"onset rows={int(onset.sum()):,} pos={D.bloom_fwd.mean():.3f}", flush=True)

            def record(df, pcol, t, model, label):
                on = df[df.chl_sonde <= df.thr_sonde]
                if len(on) < 200 or on.bloom_fwd.nunique() < 2:
                    return
                r = metrics(on.bloom_fwd, on[pcol] >= t)
                r.update(product=product, repr=rep, model=model, label=label, scope="onset", t_star=t,
                         auc=roc_auc_score(on.bloom_fwd, on[pcol]), n_pos=int(on.bloom_fwd.sum()),
                         n_stations=on.station.nunique())
                r.update(boot_ci(on, pcol, "bloom_fwd", t)); skill.append(r)
                print(f"  {model:15s} {label:9s} prec={r['precision']:.3f} pod={r['pod']:.3f} "
                      f"lift={r['lift']:.2f} [{r['lift_lo']:.2f},{r['lift_hi']:.2f}] auc={r['auc']:.3f}", flush=True)

            Dq = quantile_map(D, nar_chl)
            Dq["p"] = pack["model"].predict_proba(Dq[TIER_A].fillna(pd.Series(pack["medians"])).fillna(0.0).values)[:, 1]
            y0 = Dq.year.min(); cal = Dq[(Dq.year == y0) & onset]
            t = pick_t(cal.bloom_fwd.values, cal.p.values) if len(cal) > 50 and cal.bloom_fwd.nunique() == 2 else 0.5
            record(Dq[Dq.year > y0], "p", t, "zeroshot", "sonde")
            oof, t = rolling_refit(D, "GB")
            if oof is not None:
                record(oof, "p", t, "refit", "sonde")
            Ds = D.assign(bloom_fwd=D.y_sat).dropna(subset=["bloom_fwd"])
            oof2, t2 = rolling_refit(Ds, "GB")
            if oof2 is not None:
                oof2 = oof2.assign(chl_sonde=oof2.chl, thr_sonde=oof2.thr_sat)   # onset on the satellite's own terms
                record(oof2, "p", t2, "refit_satlabel", "satellite")
            b = rule_baseline(D)
            if b is not None: record(b, "p", 0.5, "chl_rule", "sonde")
            b = climatology_baseline(D)
            if b is not None: record(b, "p", 0.5, "climatology", "sonde")
            keys = D[["station", "date"]].assign(station=lambda x: x.station.str.split("_", n=1).str[1],
                                                 source=D.source.values)
            ups = []
            for src, sd in sonde.items():
                k = keys[keys.source == src].drop(columns="source").merge(sd, on=["station", "date"])
                if len(k):
                    ups.append(k.assign(bloom_fwd=k.y_sonde, thr=k.thr_sonde, chl_sonde=k.chl,
                                        station=src + "_" + k.station))
            if ups:
                U = pd.concat(ups).dropna(subset=["bloom_fwd"])
                oof, t = rolling_refit(U, "GB")
                if oof is not None: record(oof, "p", t, "sonde_refit", "sonde")

    pd.DataFrame(cov).to_csv("data/transfer/satellite_coverage.csv", index=False)
    pd.DataFrame(obs).to_csv("data/transfer/satellite_observability.csv", index=False)
    pd.DataFrame(agr).to_csv("data/transfer/satellite_agreement.csv", index=False)
    S = pd.DataFrame(skill); S.to_csv("data/transfer/satellite_skill.csv", index=False)

    C = pd.DataFrame(cov); O = pd.DataFrame(obs); A = pd.DataFrame(agr)
    print("\nCOVERAGE (median over stations):")
    if len(C): print(C.groupby(["product", "source"])[["frac", "frac_ffill3"]].median().round(2).to_string())
    print("\nOBSERVABILITY (share of sonde onsets with >=1 / >=3 satellite obs in prior 7 d):")
    if len(O):
        g = O.assign(w1=O.obs1 * O.onsets, w3=O.obs3 * O.onsets).groupby("product")[["w1", "w3", "onsets"]].sum()
        print(g.assign(obs1=g.w1 / g.onsets, obs3=g.w3 / g.onsets)[["onsets", "obs1", "obs3"]].round(2).to_string())
    print("\nAGREEMENT (median Spearman / kappa by product):")
    if len(A): print(A.groupby("product")[["spearman", "kappa"]].median().round(2).to_string())
    print("\nVERDICTS (obs representation):")
    for product in PRODUCTS:
        s = S[(S["product"] == product) & (S["repr"] == "obs")] if len(S) else pd.DataFrame()
        if s.empty or "refit" not in set(s.model):
            print(f"  {product}: no data"); continue
        r = s[s.model == "refit"].iloc[0]; rule = s[s.model == "chl_rule"]
        rule_l = rule.lift.iloc[0] if len(rule) else np.nan
        o = O[O["product"] == product]; ob = (o.obs1 * o.onsets).sum() / o.onsets.sum() if len(o) else np.nan
        go = (r.lift >= 1.3) and (r.lift_lo > 1.0) and (np.isnan(rule_l) or r.lift > rule_l) and (ob >= 0.6)
        print(f"  {product}: refit lift {r.lift:.2f} [{r.lift_lo:.2f},{r.lift_hi:.2f}] vs rule {rule_l:.2f}; "
              f"observability {ob:.2f} -> {'GO' if go else 'NO-GO'}")


if __name__ == "__main__":
    main()
