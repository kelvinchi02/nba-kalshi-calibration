"""Stage 10.3 machinery on small hand-made and synthetic data (no real results)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import line_movement as lm
from stats_tools import fit_logit, logit


def snapshots(prices: dict, spreads: dict | None = None) -> pd.DataFrame:
    """Long snapshot rows from {game_id: (p24, p6, p1, p0)}."""
    rows = []
    for i, (gid, ps) in enumerate(prices.items()):
        sp = (spreads or {}).get(gid, (0.01, 0.01, 0.01, 0.01))
        for h, p, s in zip((24, 6, 1, 0), ps, sp):
            rows.append({"game_id": gid, "game_date": f"2026-01-{i + 1:02d}",
                         "home_team": "AAA", "away_team": "BBB", "home_win": i % 2,
                         "horizon_h": h, "p_home": p, "max_spread": s})
    return pd.DataFrame(rows)


def test_wide_table_has_one_row_per_game_with_each_horizon():
    w = lm.wide_table(snapshots({"g1": (0.5, 0.55, 0.6, 0.62), "g2": (0.4, 0.4, 0.3, 0.2)}))
    assert w["game_id"].tolist() == ["g1", "g2"]
    assert w.loc[0, ["p_24h", "p_6h", "p_1h", "p_0h"]].tolist() == [0.5, 0.55, 0.6, 0.62]
    assert w.loc[1, "home_win"] == 1 and w.loc[1, "max_spread_0h"] == 0.01


def test_base_sample_needs_24h_and_tipoff_prices_only():
    w = lm.wide_table(snapshots({"g1": (0.5, 0.5, 0.5, 0.6), "g2": (np.nan, 0.5, 0.5, 0.6),
                                 "g3": (0.5, np.nan, 0.5, 0.6), "g4": (0.5, 0.5, 0.5, np.nan)}))
    base = lm.base_sample(w)
    assert base["game_id"].tolist() == ["g1", "g3"]
    # g3 has no 6h price: it is in the 24h model but not in the 6h model
    assert lm.model_frame(base, 24)["game_id"].tolist() == ["g1", "g3"]
    assert lm.model_frame(base, 6)["game_id"].tolist() == ["g1"]
    counts = lm.sample_counts(w, base).set_index("group")["games"]
    assert counts["main sample"] == 4 and counts["price at 24h and at tipoff"] == 2
    assert counts["of those, price at 6h"] == 1


def test_move_is_the_change_in_log_odds_of_the_home_price():
    w = lm.wide_table(snapshots({"g1": (0.5, 0.5, 0.6, 0.75), "g2": (0.8, 0.5, 0.5, 0.5)}))
    f = lm.model_frame(lm.base_sample(w), 24).set_index("game_id")
    assert f.loc["g1", "gap_move"] == pytest.approx(np.log(3))      # 0.5 -> 0.75
    assert f.loc["g2", "gap_move"] == pytest.approx(-np.log(4))     # 0.8 -> 0.5
    assert f.loc["g1", "p_home"] == 0.75                            # the tipoff price
    f1 = lm.model_frame(lm.base_sample(w), 1).set_index("game_id")
    assert f1.loc["g1", "gap_move"] == pytest.approx(logit(0.75) - logit(0.6))


def test_model_frame_refuses_a_price_of_zero_or_one():
    w = lm.wide_table(snapshots({"g1": (1.0, 0.5, 0.5, 0.6)}))
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        lm.model_frame(lm.base_sample(w), 24)


def test_spread_filter_needs_every_horizon_present_and_within_the_limit():
    w = lm.wide_table(snapshots(
        {"ok": (0.5,) * 4, "edge": (0.5,) * 4, "wide": (0.5,) * 4, "gap": (0.5,) * 4},
        {"edge": (0.05, 0.01, 0.01, 0.01), "wide": (0.01, 0.06, 0.01, 0.01),
         "gap": (0.01, 0.01, np.nan, 0.01)}))
    assert dict(zip(w["game_id"], lm.spread_ok(w))) == {
        "edge": True, "gap": False, "ok": True, "wide": False}


def synthetic(n: int = 600, d: float = 0.8, seed: int = 4) -> pd.DataFrame:
    """Games whose true log odds are logit(p0) + d * move."""
    rng = np.random.default_rng(seed)
    l24 = rng.normal(0.2, 0.7, n)
    mv = rng.normal(0, 0.4, n)
    l0 = l24 + mv
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(l0 + d * mv)))).astype(int)
    p = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return pd.DataFrame({
        "game_id": [f"g{i:04d}" for i in range(n)],
        "game_date": [date(2026, 1, 1) + timedelta(days=int(k)) for k in np.arange(n) // 6],
        "home_team": [f"T{i % 30:02d}" for i in range(n)], "away_team": "XXX",
        "home_win": y, "p_24h": p(l24), "p_6h": p(l24 + mv / 2), "p_1h": p(l24 + 0.9 * mv),
        "p_0h": p(l0),
        **{f"max_spread_{h}h": np.where(np.arange(n) % 5 == 0, 0.08, 0.02) for h in (24, 6, 1, 0)},
    })


def test_analyze_move_fits_the_plan_model_and_names_the_coefficient_d():
    base = synthetic()
    frame = lm.model_frame(base, 24)
    out = lm.analyze_move(frame, "24h", n_boot=30, seed=1)
    logi = out["logistic"].set_index("term")
    assert logi.index.tolist() == ["intercept_a", "slope_b", "move_d"]
    assert (logi["model"] == "24h").all()
    direct = fit_logit(frame["p_home"], frame["home_win"], groups=frame["home_team"],
                       extra=frame["gap_move"])
    assert logi["estimate"].to_numpy() == pytest.approx(direct.params)
    assert logi["se_cluster"].to_numpy() == pytest.approx(direct.bse)
    assert logi.loc["move_d", "estimate"] > 0.3       # the data were built with d = 0.8
    eff = out["effects"].iloc[0]
    assert eff["move_sd"] == pytest.approx(frame["gap_move"].std(ddof=1))
    assert eff["d_per_sd"] == pytest.approx(eff["move_d"] * eff["move_sd"])
    assert {"move_d", "d_per_sd", "avg_effect_per_sd"} <= set(out["bootstrap"]["statistic"])


def test_models_and_tests_table_cover_the_four_models():
    base = synthetic()
    out = lm.models(base, n_boot=20, seed=1)
    assert list(out) == ["24h", "6h", "1h", "24h_spread_filter"]
    assert out["24h_spread_filter"]["logistic"]["n"].iloc[0] == 480      # one in five dropped
    logistic = pd.concat([o["logistic"] for o in out.values()], ignore_index=True)
    tests = lm.tests_table(logistic)
    assert len(tests) == 12
    primary = tests[tests["role"] == "primary"]
    assert primary["test_id"].tolist() == ["line_move_24h_d"]
    d24 = logistic[(logistic["model"] == "24h") & (logistic["term"] == "move_d")].iloc[0]
    assert primary["p_raw"].iloc[0] == d24["p_value"]
    assert primary["boot_lo"].iloc[0] == d24["boot_lo"]
    b = tests.set_index("test_id").loc["line_move_6h_b"]
    assert b["null_hypothesis"] == "b = 1"
    assert b["statistic"] == pytest.approx((b["estimate"] - 1) / ((b["ci_hi"] - b["estimate"])
                                                                  / 1.96))


def test_move_summary_and_bins_by_hand():
    frame = pd.DataFrame({"game_id": list("abcdef"),
                          "gap_move": [-0.3, -0.1, 0.0, 0.1, 0.2, 0.7],
                          "p_home": [0.4, 0.5, 0.6, 0.5, 0.7, 0.8],
                          "home_win": [0, 1, 1, 0, 1, 1]})
    s = lm.move_summary(frame).iloc[0]
    assert s["n"] == 6 and s["mean"] == pytest.approx(0.1) and s["median"] == pytest.approx(0.05)
    assert s["share_positive"] == pytest.approx(0.5) and s["share_zero"] == pytest.approx(1 / 6)
    bins = lm.move_bins(frame, n_bins=3)
    assert bins["n"].tolist() == [2, 2, 2]
    first = bins.iloc[0]                 # moves -0.3 and -0.1, prices 0.4 and 0.5, one win
    assert first["mean_move"] == pytest.approx(-0.2)
    assert first["win_rate_minus_price"] == pytest.approx(0.5 - 0.45)
    assert first["wilson_lo_minus_price"] < first["win_rate_minus_price"] < first[
        "wilson_hi_minus_price"]
    assert bins.iloc[2]["win_rate_minus_price"] == pytest.approx(1.0 - 0.75)
