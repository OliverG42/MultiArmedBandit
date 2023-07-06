import random

from main import *


def test_interactions():
    arm_state = ArmState([0.2, 0.4, 0.5])
    assert np.all(arm_state.success_rates == np.array([1, 1, 1]))
    assert np.all(arm_state.successes == np.array([0, 0, 0]))
    assert np.all(arm_state.failures == np.array([0, 0, 0]))
    assert np.all(arm_state.arm_pulls == np.array([0, 0, 0]))
    assert arm_state.total_pulls == 0
    assert arm_state.regrets == []
    assert arm_state.num_arms == 3
    assert arm_state.reward_probs == [0.2, 0.4, 0.5]
    assert arm_state.max_prob == 0.5

    arm_state.pull_arm(0, force_result=True)
    assert np.all(arm_state.success_rates == np.array([1, 1, 1]))
    assert np.all(arm_state.successes == np.array([1, 0, 0]))
    assert np.all(arm_state.failures == np.array([0, 0, 0]))
    assert np.all(arm_state.arm_pulls == np.array([1, 0, 0]))
    assert arm_state.total_pulls == 1
    assert arm_state.regrets == [0.3]

    arm_state.pull_arm(0, force_result=True)
    arm_state.pull_arm(0, force_result=True)
    arm_state.pull_arm(0, force_result=True)
    assert np.all(arm_state.success_rates == np.array([1, 1, 1]))
    assert np.all(arm_state.successes == np.array([4, 0, 0]))
    assert np.all(arm_state.failures == np.array([0, 0, 0]))
    assert np.all(arm_state.arm_pulls == np.array([4, 0, 0]))
    assert arm_state.total_pulls == 4
    assert arm_state.regrets == [0.3, 0.3, 0.3, 0.3]

    arm_state.pull_arm(0, force_result=False)
    assert np.all(arm_state.success_rates == np.array([0.8, 1, 1]))
    assert np.all(arm_state.successes == np.array([4, 0, 0]))
    assert np.all(arm_state.failures == np.array([1, 0, 0]))
    assert np.all(arm_state.arm_pulls == np.array([5, 0, 0]))
    assert arm_state.total_pulls == 5
    assert arm_state.regrets == [0.3, 0.3, 0.3, 0.3, 0.3]

    arm_state.pull_arm(0, force_result=False)
    arm_state.pull_arm(0, force_result=True)
    assert arm_state.success_rates[0] == round(5 / 7, 4)
    assert np.all(arm_state.success_rates == np.array([round(5 / 7, 4), 1, 1]))
    assert np.all(arm_state.successes == np.array([5, 0, 0]))
    assert np.all(arm_state.failures == np.array([2, 0, 0]))
    assert np.all(arm_state.arm_pulls == np.array([7, 0, 0]))
    assert arm_state.total_pulls == 7
    assert arm_state.regrets == [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    arm_state.pull_arm(1, force_result=False)
    assert np.all(arm_state.success_rates == np.array([round(5 / 7, 4), 0, 1]))
    assert np.all(arm_state.successes == np.array([5, 0, 0]))
    assert np.all(arm_state.failures == np.array([2, 1, 0]))
    assert np.all(arm_state.arm_pulls == np.array([7, 1, 0]))
    assert arm_state.total_pulls == 8
    assert arm_state.regrets == [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1]

    arm_state.pull_arm(1, force_result=True)
    arm_state.pull_arm(1, force_result=True)
    arm_state.pull_arm(1, force_result=False)
    arm_state.pull_arm(2, force_result=False)
    arm_state.pull_arm(2, force_result=True)
    arm_state.pull_arm(2, force_result=False)
    arm_state.pull_arm(1, force_result=True)
    assert np.all(
        arm_state.success_rates == np.array([round(5 / 7, 4), 0.6, round(1 / 3, 4)])
    )
    assert np.all(arm_state.successes == np.array([5, 3, 1]))
    assert np.all(arm_state.failures == np.array([2, 2, 2]))
    assert np.all(arm_state.arm_pulls == np.array([7, 5, 3]))
    assert arm_state.total_pulls == 15
    assert arm_state.regrets == [
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.1,
        0.1,
        0.1,
        0.1,
        0.0,
        0.0,
        0.0,
        0.1,
    ]


def test_brute_force():
    arm_state = ArmState([0.2, 0.4, 0.5])

    for _ in range(0, 10000):
        arm_state.pull_arm(random.randrange(0, 2))
