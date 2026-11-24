"""Stage 9.2 machinery on small hand-made and synthetic data (no real results)."""
import numpy as np
import pandas as pd
import pytest

import horizons as hz
from stats_tools import brier, lr_test


def snapshot_rows(n_games: int = 200, seed: int = 0, noise=(0.9, 0.5, 0.2, 0.0)) -> pd.DataFrame:
    """Synthetic games. The tipoff price is calibrated; earlier prices add logit noise."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.1, 0.9, n_games)
    win = (rng.uniform(size=n_games) < truth).astype(int)
    rows = []
    for i in range(n_games):
        for h, sd in zip(hz.HORIZONS, noise):
            logit_p = np.log(truth[i] / (1 - truth[i])) + rng.normal(0, sd) if sd else \
                np.log(truth[i] / (1 - truth[i]))
            rows.append({"game_id": f"{i:04d}", "game_date": pd.Timestamp("2025-10-21").date()
                         + pd.Timedelta(days=i // 8), "home_team": f"T{i % 30:02d}",
                         "away_team": f"T{(i + 7) % 30:02d}", "horizon_h": h,
                         "p_home": 1 / (1 + np.exp(-logit_p)), "home_win": win[i],
                         "max_spread": 0.02, "price_source": "mid"})
    return pd.DataFrame(rows)


def test_complete_games_drops_a_game_from_every_horizon_when_one_price_is_missing():
    sn = snapshot_rows(20)
    sn.loc[(sn["game_id"] == "0003") & (sn["horizon_h"] == 24), "p_home"] = np.nan
    d = hz.complete_games(sn)
    assert d["game_id"].nunique() == 19 and "0003" not in set(d["game_id"])
    assert (d.groupby("horizon_h").size() == 19).all()       # same games at every horizon
    counts = hz.sample_counts(sn, d).set_index("group")["games"]
    assert counts["main sample"] == 20
    assert counts["p_home present at 24h"] == 19 and counts["p_home present at tipoff"] == 20
    assert counts["p_home present at all four horizons"] == 19


def test_complete_games_rejects_a_price_of_exactly_0_or_1():
    sn = snapshot_rows(10)
    sn.loc[0, "p_home"] = 1.0
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        hz.complete_games(sn)


def test_price_source_counts():
    sn = snapshot_rows(10)
    sn.loc[(sn["horizon_h"] == 24) & (sn["game_id"] < "0003"), "price_source"] = "last"
    out = hz.price_source_counts(hz.complete_games(sn)).set_index("horizon_h")
    assert out.index.tolist() == [24, 6, 1, 0]
    assert out.loc[24, "last"] == 3 and out.loc[24, "mid"] == 7 and out.loc[0, "last"] == 0


def test_spread_filter_drops_whole_games():
    d = hz.complete_games(snapshot_rows(12))
    d.loc[(d["game_id"] == "0001") & (d["horizon_h"] == 6), "max_spread"] = 0.06
    d.loc[(d["game_id"] == "0002") & (d["horizon_h"] == 0), "max_spread"] = np.nan
    d.loc[(d["game_id"] == "0004") & (d["horizon_h"] == 1), "max_spread"] = 0.05  # at the limit
    kept = hz.spread_filter(d)
    assert set(d["game_id"]) - set(kept["game_id"]) == {"0001", "0002"}
    assert (kept.groupby("horizon_h").size() == 10).all()


def test_horizon_metrics_match_the_stage_8_functions():
    d = hz.complete_games(snapshot_rows(400, seed=1))
    m = hz.horizon_metrics(d, "all").set_index("horizon_h")
    assert m.index.tolist() == [24, 6, 1, 0] and (m["sample"] == "all").all()
    g = d[d["horizon_h"] == 6]
    assert m.loc[6, "brier"] == brier(g["p_home"], g["home_win"])["brier"]
    assert m.loc[6, "lr_p_value"] == lr_test(g["p_home"], g["home_win"])["p_value"]
    assert m["brier_baseline"].nunique() == 1                 # same games, same baseline
    assert (m["n"] == 400).all()
    assert m.loc[24, "brier"] > m.loc[0, "brier"]             # noisier prices score worse


def test_paired_bootstrap_scores_every_horizon_on_the_same_resample():
    d = hz.complete_games(snapshot_rows(240, seed=2))
    boot = hz.paired_brier_bootstrap(d, n_boot=30, seed=5)
    again = hz.paired_brier_bootstrap(d, n_boot=30, seed=5)
    pd.testing.assert_frame_equal(boot, again)
    for h in hz.EARLIER:
        assert np.allclose(boot[f"brier_diff_{h}h_minus_0h"],
                           boot[f"brier_{h}h"] - boot["brier_0h"])
    # identical prices at two horizons give a difference of exactly 0 in every replicate
    same = d.copy()
    tip = same[same["horizon_h"] == 0].set_index("game_id")["p_home"]
    mask = same["horizon_h"] == 1
    same.loc[mask, "p_home"] = same.loc[mask, "game_id"].map(tip).to_numpy()
    assert (hz.paired_brier_bootstrap(same, n_boot=10, seed=5)["brier_diff_1h_minus_0h"] == 0).all()


def test_bootstrap_and_diff_tables():
    d = hz.complete_games(snapshot_rows(240, seed=3))
    boot = hz.paired_brier_bootstrap(d, n_boot=50, seed=5)
    summary = hz.bootstrap_table(d, boot, "all")
    assert len(summary) == 7 and (summary["n_dates"] == d["game_date"].nunique()).all()
    diff = hz.brier_diff_table(summary).set_index("horizon_h")
    assert diff.index.tolist() == [24, 6, 1]
    g24, g0 = d[d["horizon_h"] == 24], d[d["horizon_h"] == 0]
    expected = brier(g24["p_home"], g24["home_win"])["brier"] - brier(g0["p_home"], g0["home_win"])["brier"]
    assert diff.loc[24, "brier_diff"] == pytest.approx(expected)
    assert diff.loc[24, "brier_diff"] == pytest.approx(diff.loc[24, "brier_h"] - diff.loc[24, "brier_0h"])
    assert set(diff["interval_excludes_0"]) <= {0, 1}


def test_calibration_bins_per_horizon():
    d = hz.complete_games(snapshot_rows(200, seed=4))
    bins = hz.horizon_calibration_bins(d)
    assert len(bins) == 40 and (bins.groupby("horizon_h")["n"].sum() == 200).all()
    assert (bins["n"] == 20).all()
    one = bins[bins["horizon_h"] == 24]
    assert one["mean_p"].is_monotonic_increasing             # binned on that horizon's prices
    assert (bins["wilson_lo"] <= bins["win_rate"]).all() and (bins["win_rate"] <= bins["wilson_hi"]).all()


def test_labels():
    assert [hz.label(h) for h in hz.HORIZONS] == ["24h", "6h", "1h", "tipoff"]
    assert set(hz.HORIZON_COLORS) == set(hz.HORIZONS)
