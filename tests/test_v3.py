import itertools
import pytest
from rewards.v3 import ARMS,reward_group,verify_final
from evaluation.answer_contracts import verify_formatted_final


def test_formatted_answer_is_secondary_and_has_no_intermediate_fallback():
    for text in ('Answer: 50.','**Answer:** 50','Answer: **50**','Answer: 50.0.'):
        assert verify_formatted_final(text,'50')[0]
    for text in ('Angle is 50','Answer: 50+2','Answer: 50/2','Answer: 50 units','Answer: 50.2'):
        assert not verify_formatted_final(text,'50')[0]


def test_final_contract_rejects_truncated_and_intermediate_numbers():
    assert verify_final('Thus 8. Answer: 8','8')[0]
    assert verify_final('Answer: 8.0','8')[0]
    assert not verify_final('Try x = 4: 4','4')[0]
    assert not verify_final('Answer: 4+2','4')[0]
    assert not verify_final('Answer: 4/2','4')[0]
    assert not verify_final('Answer: nan','0')[0]


def test_corrected_zero_sum_and_failure_invariance():
    for passes in itertools.product((False,True),repeat=4):
        for tokens in ([1,2,3,9],[10]*4,[100,11,27,320]):
            for arm in ARMS:
                r=reward_group(list(passes),tokens,arm)
                assert all(r['rewards'][i]==0 and r['delta'][i]==0 for i,p in enumerate(passes) if not p)
                if arm in ('centered_budget','centered_fixed'):assert abs(r['delta_sum'])<1e-12


def test_center_correction_changes_only_three_success_groups_at_g4():
    for k in range(5):
        passes=[True]*k+[False]*(4-k)
        a=reward_group(passes,[10,20,70,100],'legacy_budget')
        b=reward_group(passes,[10,20,70,100],'centered_budget')
        if k!=3:assert a['normalized_advantage']==pytest.approx(b['normalized_advantage'],abs=1e-12)
        else:assert a['normalized_advantage']!=pytest.approx(b['normalized_advantage'])


def test_grs_is_present_in_all_redistribution_arms():
    for arm in ('legacy_budget','centered_budget','centered_fixed'):
        r=reward_group([True,False,False,False],[100]*4,arm)
        g=reward_group([True,False,False,False],[100]*4,'grs_only')
        assert r['rewards']==g['rewards']


def test_published_rescaling_adaptation_and_all_success_filter():
    r=reward_group([True,True,False,False],[10,30,20,20],'gr3_fixed')
    assert r['rewards']==pytest.approx([1/(1+.33*.5),1/(1+.33*1.5),0,0])
    assert reward_group([True]*4,[10,20,30,40],'gr3_fixed')['normalized_advantage']==[0]*4


def test_mixed_group_sign_bound_and_fixed_coefficient_counterexample():
    for k in (1,2,3):
        for costs in itertools.product((1,24,384,768),repeat=k):
            passed=[True]*k+[False]*(4-k)
            r=reward_group(passed,list(costs)+[768]*(4-k),'centered_budget')
            assert all(x>0 for x in r['normalized_advantage'][:k])
    adaptive=reward_group([True,True,True,False],[1,1,768,500],'centered_budget')
    fixed=reward_group([True,True,True,False],[1,1,768,500],'centered_fixed')
    assert adaptive['normalized_advantage'][2]>0
    assert fixed['normalized_advantage'][2]<0
