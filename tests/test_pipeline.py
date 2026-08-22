"""Tests for the eval pipeline: resolve per IP, estimate, evaluate."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimators.baselines import BASELINES
from eval import pipeline

# Anchor in Berlin; three sources: two close by, one grossly off (London).
ANCHORS = [{"ip": "1.2.3.4", "lat": "52.52", "lon": "13.40",
            "city": "Berlin", "country": "DE", "asn": "3320"}]
OBSERVATIONS = [
    {"ip": "1.2.3.4", "source": "ip_api", "lat": "52.50", "lon": "13.41",
     "status": "success", "fetched_at_utc": "2026-06-03T00:00:00Z"},
    {"ip": "1.2.3.4", "source": "ipwho_is", "lat": "52.53", "lon": "13.39",
     "status": "success", "fetched_at_utc": "2026-06-03T00:00:00Z"},
    {"ip": "1.2.3.4", "source": "ipapi_co", "lat": "51.51", "lon": "-0.13",
     "status": "success", "fetched_at_utc": "2026-06-03T00:00:00Z"},
    # failed source is ignored
    {"ip": "1.2.3.4", "source": "broken", "lat": "", "lon": "",
     "status": "http_error", "fetched_at_utc": "2026-06-03T00:00:00Z"},
]


def test_load_cases_filters_and_builds_points():
    cases = pipeline.load_cases(ANCHORS, OBSERVATIONS)
    assert len(cases) == 1
    case = cases[0]
    assert case["truth"] == (52.52, 13.40)
    assert case["points"].shape == (3, 2)          # http_error source dropped
    assert set(case["sources"]) == {"ip_api", "ipwho_is", "ipapi_co"}
    assert len(case["provenance"]) == 3            # RQ4: traceability


def test_anchor_without_observations_dropped():
    cases = pipeline.load_cases(
        ANCHORS + [{"ip": "9.9.9.9", "lat": "0", "lon": "0"}], OBSERVATIONS)
    assert {c["ip"] for c in cases} == {"1.2.3.4"}


def test_evaluate_robust_beats_centroid():
    cases = pipeline.load_cases(ANCHORS, OBSERVATIONS)
    df = pipeline.evaluate(cases, BASELINES, include_sources=True)
    # per estimator one aggregate row + three source rows
    agg = df[df["kind"] == "aggregate"].set_index("estimator")["error_km"]
    assert set(agg.index) == set(BASELINES)
    assert (df["kind"] == "source").sum() == 3
    # the London outlier pulls the centroid much more strongly than the geometric median
    assert agg["centroid"] > agg["geometric_median"]
    assert (df["n_sources"] == 3).all()


def test_line_weights_split_per_lineage():
    import numpy as np
    # two sources share one line, one has its own line
    w = pipeline.line_weights(["maxmind", "maxmind", "own"])
    assert np.allclose(w, [0.5, 0.5, 1.0])


def test_evaluate_line_weighted_column(monkeypatch):
    from estimators.baselines import weighted_geometric_median
    # attach lineage to the observations: ip_api its own line, the other two shared
    obs = [dict(o) for o in OBSERVATIONS if o["status"] == "success"]
    lineage = {"ip_api": "ipapi", "ipwho_is": "shared", "ipapi_co": "shared"}
    for o in obs:
        o["lineage"] = lineage[o["source"]]
    cases = pipeline.load_cases(ANCHORS, obs)
    df = pipeline.evaluate(cases, {}, line_weighted={"gm_perline": weighted_geometric_median})
    assert "gm_perline" in set(df["estimator"])
    assert (df["n_lines"] == 2).all()   # ipapi + shared
