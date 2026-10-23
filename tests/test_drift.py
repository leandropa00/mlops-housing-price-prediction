import numpy as np
from housing_mlops.drift import population_stability_index


def test_identical_distribution_is_zero():
    x=np.linspace(1,10,200)
    assert abs(population_stability_index(x,x))<1e-8


def test_large_shift_triggers_alert():
    x=np.linspace(1,10,200)
    assert population_stability_index(x,x+7)>0.2
