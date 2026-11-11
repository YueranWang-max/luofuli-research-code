"""Numerical invariants for the secondary uncertainty calculation."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('secondary',Path(__file__).resolve().parents[1]/'paper/analyze_v3_secondary.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_strata_keep_opposing_constant_difficulties_balanced():
    # Unstratified resampling would incorrectly create uncertainty in this construction.
    d=np.tile([1.,1.,-1.,-1.],(5,1))
    result=module.bootstrap_only(d,['L4','L4','L5','L5'])
    assert result['difference']==0
    assert result['conditional_item_ci95']==[0.,0.]
    assert result['crossed_seed_item_ci95']==[0.,0.]


def test_crossed_interval_retains_training_seed_variation():
    d=np.repeat(np.array([-1.,0.,0.,0.,1.])[:,None],4,axis=1)
    result=module.bootstrap_only(d,['L4','L4','L5','L5'])
    assert result['conditional_item_ci95']==[0.,0.]
    assert result['crossed_seed_item_ci95'][0]<0<result['crossed_seed_item_ci95'][1]
    assert result['seed_differences']==[-1.,0.,0.,0.,1.]


def test_uniform_difference_is_not_rescaled_by_number_of_seeds():
    result=module.bootstrap_only(np.full((5,4),.5),['L4']*4)
    assert result['difference']==.5
    assert result['crossed_seed_item_ci95']==[.5,.5]
