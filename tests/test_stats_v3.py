import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper'))
from stats_v3 import holm,paired_effect

def test_holm_preserves_original_order_and_caps_at_one():
    assert holm([.04,.001,.02,.8])==pytest.approx([.08,.004,.06,.8])
    with pytest.raises(ValueError):holm([float('nan')])

def test_zero_difference_and_constant_effect_preserve_pairing():
    zero=paired_effect(np.zeros((5,12)),[0]*6+[1]*6,permutations=100,bootstraps=100)
    assert zero['conditional_item_permutation_p']==1
    assert zero['crossed_seed_item_ci95']==[0,0]
    constant=paired_effect(np.full((5,12),.5),[0]*6+[1]*6,permutations=100,bootstraps=100)
    assert constant['conditional_item_ci95']==[.5,.5]
    assert constant['crossed_seed_item_ci95']==[.5,.5]

def test_seed_variation_is_not_erased_by_item_resampling():
    d=np.repeat(np.array([-.4,-.2,0,.2,.4])[:,None],20,axis=1)
    result=paired_effect(d,[0]*10+[1]*10,permutations=100,bootstraps=1000)
    assert result['conditional_item_ci95']==pytest.approx([0,0],abs=1e-15)
    assert result['crossed_seed_item_ci95'][0]<0<result['crossed_seed_item_ci95'][1]
