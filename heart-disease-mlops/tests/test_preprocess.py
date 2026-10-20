from src.preprocess import load_heart_data

def test_target_is_binary():
    df = load_heart_data()
    assert set(df["target"].unique()) == {0, 1}

def test_no_missing_after_load():
    df = load_heart_data()
    assert df.isna().sum().sum() == 0

def test_row_count():
    df = load_heart_data()
    assert len(df) == 297  # 303 - 6 missing