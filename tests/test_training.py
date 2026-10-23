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
    # Criterio de negocio: no basta con que el entrenamiento finalice.
    baseline_mae = float(df.loc[df.experiment == "baseline", "mae"].iloc[0])
    best_mae = float(df["mae"].min())
    assert best_mae <= 4.0, f"MAE fuera del objetivo: {best_mae}"
    assert 1 - best_mae / baseline_mae >= 0.25, (
        f"Mejora insuficiente frente al baseline: {baseline_mae=}, {best_mae=}"
    )
    assert abs(result["mae"] - best_mae) < 1e-5
