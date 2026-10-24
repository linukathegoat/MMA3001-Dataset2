"""Independent, reproducible unit tests (no Monash raw data needed)."""
import numpy as np
import pandas as pd
import pytest
from src.sensor_project import (predict,select_masked_blocks,evaluate,
                                choose_by_validation_mae)


def test_linear_interpolation_known_answer():
    x=np.array([0,10,20,30],dtype=float)
    y=np.array([20,22,24,26],dtype=float)
    np.testing.assert_allclose(predict("linear",x,y,np.array([5,15])),[21,23])


def test_last_known_is_previous_reading():
    np.testing.assert_allclose(predict("last_known",np.array([0,10,20.]),
                                       np.array([20,22,24.]),np.array([5,15])),[20,22])


@pytest.mark.parametrize("method",["linear","pchip","cubic_spline"])
def test_constant_temperature_stays_constant(method):
    x=np.array([0.,10.,20.,30.,40.]);y=np.ones(5)*23.4
    np.testing.assert_allclose(predict(method,x,y,np.array([5.,15.,25.])),23.4,atol=1e-10)


def test_extrapolation_is_refused():
    with pytest.raises(ValueError,match="extrapolation"):
        predict("linear",np.array([0.,10.,20.]),np.array([20.,21.,22.]),np.array([-1.]))


def test_duplicate_times_are_refused():
    with pytest.raises(ValueError,match="sorted, unique"):
        predict("pchip",np.array([0.,10.,10.]),np.array([20.,21.,22.]),np.array([5.]))


def test_masks_are_reproducible_disjoint_and_keep_neighbors():
    x=np.arange(350,dtype=float)*600
    blocks=select_masked_blocks(x,6,n_blocks=15,seed=4)
    blocks2=select_masked_blocks(x,6,n_blocks=15,seed=4)
    assert [list(a) for a in blocks]==[list(a) for a in blocks2]
    values=np.concatenate(blocks)
    assert len(values)==len(np.unique(values))==90
    for b in blocks:
        assert b[0]>1 and b[-1]<len(x)-2


def test_masks_reject_existing_long_sampling_gaps():
    x=np.arange(200,dtype=float)*600
    x[100:]+=86400
    blocks=select_masked_blocks(x,6,n_blocks=10,seed=12)
    for b in blocks:
        assert not(b[0]<=100<=b[-1]+1)


def test_full_evaluation_does_not_use_hidden_readings():
    x=np.arange(500,dtype=float)*600
    y=22+0.8*np.sin(np.arange(500)/19)
    df=pd.DataFrame({"unix_seconds":x,"temperature_c":y})
    s,p,_=evaluate(df,gap_lengths=(3,),n_blocks=8,seed=47)
    assert len(s)==4
    assert set(p.method)=={"last_known","linear","pchip","cubic_spline"}
    assert len(p)==8*3*4
    assert choose_by_validation_mae(s) in set(p.method)
