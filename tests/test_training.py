from pathlib import Path
import pandas as pd
from housing_mlops.training import FEATURES, BASE, train


def test_training_three_experiments_and_artifact(tmp_path):
    result=train(dataset=BASE/"data/housing.csv",output=tmp_path,tracking=False)
    assert result["run_count"]==3
    assert (tmp_path/"model.joblib").is_file()
    df=pd.read_csv(tmp_path/"experiments.csv")
    assert set(df.experiment)=={"baseline","ridge","forest"}
    assert not df[["mae","rmse","r2"]].isna().any().any()
    assert "b" not in FEATURES
