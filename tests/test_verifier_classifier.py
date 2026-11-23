from scripts.train_verifier_classifier import compose_prediction, macro_f1


def test_compose_prediction_enforces_protocol_steps():
    assert compose_prediction(True, 0, 3, 5) == "valid"
    assert compose_prediction(False, 0, 1, 5) == (
        "invalid | step=5 | reason=goal_not_reached"
    )
    assert compose_prediction(False, 1, 9, 5) == (
        "invalid | step=4 | reason=obstacle_collision"
    )


def test_macro_f1_uses_both_fixed_labels():
    assert macro_f1([0, 1], [0, 1]) == 1.0
    assert macro_f1([0, 1], [1, 1]) == 1 / 3
