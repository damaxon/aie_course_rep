from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename" : ["1.jpg","2.jpg","3.jpg","3.jpg"],
            "width": [512, 512, 512, 512],
            "height": [512, 512, 512, 512],
            "class": ["car","car","pedestrian",None],
            "xmin":[291,270,0,25],
            "ymin":[247,235,266,258],
            "xmax":[520,293,13,106],
            "ymax":[331,321,327,304],
        }
    )




def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 8
    assert any(c.name == "filename" for c in summary.columns)
    assert any(c.name == "width" for c in summary.columns)
    assert any(c.name == "height" for c in summary.columns)
    assert any(c.name == "class" for c in summary.columns)
    assert any(c.name == "xmin" for c in summary.columns)
    assert any(c.name == "ymin" for c in summary.columns)
    assert any(c.name == "xmax" for c in summary.columns)
    assert any(c.name == "ymax" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["class", "missing_count"] == 1


def test_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)
    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)

    assert 0.0 <= flags["quality_score"] <= 1.0
    assert flags["has_suspicious_filename_duplicates"]==True
    assert flags["has_constant_columns"] == True
    assert flags["has_invalid_coord_bounding_box"] == True
    assert flags["has_invalid_image_resolution"] == False
    assert flags["too_few_rows"] == True
    assert flags["too_many_columns"] == False


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "class" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "class" in top_cats
    city_table = top_cats["class"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2
