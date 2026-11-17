# Member C stage-acceptance cases

All model-like outputs below are deterministic mocks; the final checks use A's real Executor.

## Successful repair

- sample: `id_d83d8b6aedae036977d87b99875c80bc368ce1a37a4e04818016b88ad0287a1b`
- corruption: `action_substitution`
- candidate: `['up', 'right', 'right', 'right']`
- feedback: `{'valid': False, 'executor_valid': True, 'success': False, 'first_error_step': 4, 'reason': 'goal_not_reached', 'prediction_kind': 'executor_ground_truth_adapter'}`
- repaired: `['right', 'right', 'right', 'right']`
- final success: `True`

## Preserved failure

- sample: `id_ebcdecc0f65d0d7cdc3b7f0b6a40151d7b3c28d524e34c460fdcaf698fbc7129`
- corruption: `premature_stopping`
- candidate: `['up', 'right', 'up']`
- feedback: `{'valid': False, 'executor_valid': True, 'success': False, 'first_error_step': 3, 'reason': 'goal_not_reached', 'prediction_kind': 'executor_ground_truth_adapter'}`
- repaired: `['up', 'right', 'up']`
- final result: `{'valid': True, 'success': False, 'first_error_step': 3, 'reason': 'goal_not_reached', 'final_position': [3, 3], 'executed_steps': 3, 'trace': [[5, 2], [4, 2], [4, 3], [3, 3]], 'goal_not_reached_step_convention': 'len(plan) (0-based)'}`
- explanation: deterministic failure injection verifies that an unsuccessful repair remains visible and is counted as failure.
