"""Trace-only Generation format. No path search or executor selection."""
import re
from collections import Counter

DELTAS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
ACTION_RE = re.compile(r'(?:up|down|left|right)(?:, (?:up|down|left|right))*')
COORD = r'\(([0-9]+),([0-9]+)\)'
TRACE_RE = re.compile(COORD + r'(?: (?:up|down|left|right) ' + COORD + r')+')


def map_text(grid):
    obstacles = {tuple(x) for x in grid['obstacles']}
    lines = ['Map legend: S=start, G=goal, X=obstacle, .=free.',
             'Columns: ' + ' '.join(str(c) for c in range(grid['cols']))]
    for r in range(grid['rows']):
        cells = []
        for c in range(grid['cols']):
            cell = [r, c]
            symbol = 'S' if cell == grid['start'] else 'G' if cell == grid['goal'] else 'X' if (r, c) in obstacles else '.'
            cells.append(symbol)
        lines.append(f'Row {r}: ' + ' '.join(cells))
    return '\n'.join(lines)


def build_trace_prompt(g):
    return ('Find a path that reaches the goal without crossing obstacles or leaving the grid.\n'
              'Coordinates are (row,col), starting at (0,0) in the top left.\n'
              f"Grid size: {g['rows']} rows, {g['cols']} columns.\n"
              f"Start: ({g['start'][0]},{g['start'][1]}). Goal: ({g['goal'][0]},{g['goal'][1]}).\n"
              f"Obstacles: {g['obstacles']}.\n" + map_text(g) + '\n'
              'up decreases row; down increases row; left decreases column; right increases column.\n'
              'Output exactly one Trace: line with the start coordinate, then each action and its resulting coordinate. '
              'Use coordinates formatted (row,col). Stop the path at the goal.\nAnswer:')


def make_example(row, mode):
    if mode != 'trace_only':
        raise ValueError('Expected trace_only mode')
    g = row['grid']
    prompt = build_trace_prompt(g)
    if not ACTION_RE.fullmatch(row['target_text']):
        raise ValueError(f"Unexpected reference actions: {row['base_sample_id']}")
    actions = row['target_text'].split(', ')
    r, c = g['start']
    parts = [f'({r},{c})']
    for action in actions:
        dr, dc = DELTAS[action]
        r, c = r + dr, c + dc
        parts.extend([action, f'({r},{c})'])
    if [r, c] != g['goal']:
        raise ValueError('Reference actions do not end at the goal')
    return {**row, 'original_input_text': row['input_text'], 'reference_actions': actions,
            'reference_action_text': row['target_text'], 'input_text': prompt,
            'target_text': 'Trace: ' + ' '.join(parts)}


def parse_output(raw, mode, grid):
    if mode != 'trace_only':
        raise ValueError('Expected trace_only mode')
    result = dict(parse_success=False, plan=[], trace_parse_success=False,
                  trace_actions_agree=None, trace_transition_consistent=None,
                  trace_start_matches=None, trace_end_matches_goal=None)
    match = re.fullmatch(r'Trace:\s*(.*?)', raw.strip(), flags=re.DOTALL)
    if match is None or not TRACE_RE.fullmatch(match.group(1)):
        return result
    body = match.group(1)
    positions = [[int(r), int(c)] for r, c in re.findall(COORD, body)]
    actions = re.findall(r'\b(up|down|left|right)\b', body)
    consistent = all([positions[i][0] + DELTAS[a][0], positions[i][1] + DELTAS[a][1]] == positions[i + 1]
                     for i, a in enumerate(actions))
    start_matches = positions[0] == grid['start']
    valid = consistent and start_matches
    result.update(parse_success=valid, plan=actions if valid else [], trace_parse_success=True,
                  trace_positions=positions, trace_actions=actions,
                  trace_transition_consistent=consistent, trace_start_matches=start_matches,
                  trace_end_matches_goal=positions[-1] == grid['goal'])
    return result


def diagnostic_execute(grid, actions):
    """Local movement check for analysis; A's executor remains authoritative."""
    r, c = grid['start']
    obstacles = {tuple(cell) for cell in grid['obstacles']}
    for action in actions:
        dr, dc = DELTAS[action]
        r, c = r + dr, c + dc
        if not (0 <= r < grid['rows'] and 0 <= c < grid['cols']):
            return 'out_of_bounds'
        if (r, c) in obstacles:
            return 'obstacle_collision'
    return 'success' if [r, c] == grid['goal'] else 'goal_not_reached'


def summarize(predictions, mode):
    if mode != 'trace_only':
        raise ValueError('Expected trace_only mode')
    n = len(predictions)
    result = dict(count=n, action_parse_rate=sum(p['parse_success'] for p in predictions) / n,
                  action_exact_match_rate=sum(p['action_exact_match'] for p in predictions) / n,
                  unique_action_predictions=len({tuple(p['plan']) for p in predictions if p['parse_success']}),
                  hit_generation_limit_count=sum(p['hit_generation_limit'] for p in predictions),
                  most_common_outputs=Counter(p['raw_output'] for p in predictions).most_common(3),
                  diagnostic_success_rate=sum(p['diagnostic_reason'] == 'success' for p in predictions) / n,
                  diagnostic_failure_reasons=dict(Counter(p['diagnostic_reason'] for p in predictions)),
                  official_success_rate=None,
                  note='Reference equality is a proxy; official A executor evaluation pending.')
    for key in ('trace_parse_success', 'trace_transition_consistent',
                'trace_start_matches', 'trace_end_matches_goal'):
        result[key + '_rate'] = sum(p['parsed'][key] is True for p in predictions) / n
    return result
