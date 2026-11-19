"""Re-score existing v7 grid_state predictions using Trace as the action source.

CPU-only postprocessing. The local movement check is diagnostic, not A's official executor.
No candidate selection, path search, or correction is performed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from state_format import ACTION_RE, diagnostic_execute, parse_output


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path, required=True,
                    help='v7 grid_state dev_predictions.jsonl')
    ap.add_argument('--output', type=Path, required=True,
                    help='New, nonexisting output directory')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing overwrite: {args.output}')
    rows = read_rows(args.predictions)
    if len(rows) != 100:
        raise ValueError(f'Expected fixed Dev100 predictions, found {len(rows)}')
    if len({row['base_sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    result = []
    for row in rows:
        if row.get('mode') != 'grid_state' or row.get('task') != 'generation':
            raise ValueError('Expected v7 grid_state Generation predictions')
        raw = row['raw_output'].strip()
        trace_text, action_text = '', ''
        if raw.startswith('Trace:'):
            body = raw[len('Trace:'):].strip()
            marker = re.search(r'\s+Actions:\s*', body)
            if marker:
                trace_text, action_text = body[:marker.start()], body[marker.end():]
            else:
                trace_text = body
        baseline_valid = bool(ACTION_RE.fullmatch(action_text))
        baseline_plan = action_text.split(', ') if baseline_valid else []
        parsed_trace = parse_output('Trace: ' + trace_text, 'trace_only', row['grid'])
        trace_valid = parsed_trace['parse_success']
        trace_plan = parsed_trace['plan']
        reference = row['target_text'].split(', ')
        baseline_reason = diagnostic_execute(row['grid'], baseline_plan) if baseline_valid else 'parse_failure'
        trace_reason = diagnostic_execute(row['grid'], trace_plan) if trace_valid else 'parse_failure'
        result.append(dict(task='generation', base_sample_id=row['base_sample_id'],
                           environment_id=row['environment_id'], grid=row['grid'],
                           reference_action_text=row['target_text'], raw_output=row['raw_output'],
                           original_actions_plan=baseline_plan, original_actions_parse_success=baseline_valid,
                           original_actions_exact_match=baseline_valid and baseline_plan == reference,
                           original_actions_diagnostic_reason=baseline_reason,
                           trace_plan=trace_plan, trace_parse_success=trace_valid,
                           plan=trace_plan, parse_success=trace_valid,
                           candidate_plan=trace_plan if trace_valid else None,
                           trace_exact_match=trace_valid and trace_plan == reference,
                           trace_diagnostic_reason=trace_reason,
                           trace_coordinates_consistent=parsed_trace['trace_transition_consistent'],
                           trace_start_matches=parsed_trace['trace_start_matches'],
                           trace_end_matches_goal=parsed_trace['trace_end_matches_goal'],
                           plans_agree=baseline_valid and trace_valid and baseline_plan == trace_plan))
    n = len(result)
    report = dict(source=str(args.predictions), count=n,
                  note='Same saved predictions, no retraining or candidate selection. Diagnostic movement check only; official A success pending.',
                  original_actions=dict(parse_count=sum(r['original_actions_parse_success'] for r in result),
                                        exact_match_count=sum(r['original_actions_exact_match'] for r in result),
                                        diagnostic_success_count=sum(r['original_actions_diagnostic_reason'] == 'success' for r in result),
                                        failure_reasons=dict(Counter(r['original_actions_diagnostic_reason'] for r in result))),
                  trace=dict(parse_count=sum(r['trace_parse_success'] for r in result),
                             exact_match_count=sum(r['trace_exact_match'] for r in result),
                             diagnostic_success_count=sum(r['trace_diagnostic_reason'] == 'success' for r in result),
                             failure_reasons=dict(Counter(r['trace_diagnostic_reason'] for r in result))),
                  both_succeed_count=sum(r['original_actions_diagnostic_reason'] == 'success' and r['trace_diagnostic_reason'] == 'success' for r in result),
                  trace_only_succeeds_count=sum(r['original_actions_diagnostic_reason'] != 'success' and r['trace_diagnostic_reason'] == 'success' for r in result),
                  actions_only_succeeds_count=sum(r['original_actions_diagnostic_reason'] == 'success' and r['trace_diagnostic_reason'] != 'success' for r in result),
                  plans_agree_count=sum(r['plans_agree'] for r in result),
                  official_success_rate=None)
    args.output.mkdir(parents=True)
    write_json(args.output / 'trace_comparison_report.json', report)
    (args.output / 'trace_predictions.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in result), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
