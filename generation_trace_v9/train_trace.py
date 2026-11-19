"""Offline Flan-T5-base: trace-only output to remove duplicate action copying.

Same frozen Train2000/Dev100 IDs, three epochs, fixed seed42. No A executor bundled.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'generation_models_v5'))
from model_assets import verify_model
from state_format import diagnostic_execute, make_example, parse_output, summarize


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def input_hash(rows):
    return hashlib.sha256(json.dumps([[r['base_sample_id'], r['input_text'], r['target_text']] for r in rows],
                                    ensure_ascii=False).encode()).hexdigest()


def load_data(data_dir, dev_dir):
    summary = json.loads((data_dir / 'summary.json').read_text(encoding='utf-8'))
    dev_summary = json.loads((dev_dir / 'summary.json').read_text(encoding='utf-8'))
    for directory, metadata in ((data_dir, summary), (dev_dir, dev_summary)):
        for name, expected in metadata['file_sha256'].items():
            if digest(directory / name) != expected:
                raise ValueError(f'Frozen file changed: {directory / name}')
    train = read_rows(data_dir / 'generation_train.jsonl')
    original_dev = read_rows(data_dir / 'generation_dev.jsonl')
    monitor = read_rows(data_dir / 'monitor_train.jsonl')
    dev = read_rows(dev_dir / 'generation_dev.jsonl')
    if len(train) != 2000 or len(dev) != 100 or len(original_dev) != 20 or len(monitor) != 200:
        raise ValueError('Expected original Train2000, Dev100 and fixed Train monitor200')
    if dev[:20] != original_dev or input_hash(original_dev) != dev_summary['old_dev_input_sha256']:
        raise ValueError('Original Dev20 not preserved')
    if input_hash(train) != summary['train_model_input_sha256']:
        raise ValueError('Train input/target hash mismatch')
    if summary['size'] != 2000 or dev_summary['task'] != 'generation':
        raise ValueError('Wrong dataset summary')
    train_by_id = {r['base_sample_id']: r for r in train}
    if len(train_by_id) != 2000 or len({r['base_sample_id'] for r in dev}) != 100:
        raise ValueError('Duplicate sample IDs')
    if len({r['base_sample_id'] for r in monitor}) != 200 or any(train_by_id.get(r['base_sample_id']) != r for r in monitor):
        raise ValueError('Monitor is not the unchanged original training subset')
    heldout = set(json.loads((data_dir / 'heldout_environment_ids.json').read_text(encoding='utf-8')))
    allowed_dev = set(json.loads((dev_dir / 'dev_environment_ids.json').read_text(encoding='utf-8')))
    train_env, dev_env = ({r['environment_id'] for r in subset} for subset in (train, dev))
    if train_env & dev_env or (train_env | dev_env) & heldout or not dev_env <= allowed_dev:
        raise ValueError('Environment split violation')
    for split, subset in (('train', train), ('dev', dev)):
        for row in subset:
            if row['split'] != split or row['task'] != 'generation' or row['grid']['environment_id'] != row['environment_id']:
                raise ValueError('Unexpected row identity')
            if (row['grid']['rows'], row['grid']['cols']) != (6, 6):
                raise ValueError('Expected 6x6 maps')
    return train, dev, monitor, summary, dev_summary


def encode(rows, tokenizer, target_limit):
    features, lengths = [], {'max_input_tokens': 0, 'max_target_tokens': 0}
    for row in rows:
        source = tokenizer(row['input_text'], truncation=False)
        labels = tokenizer(text_target=row['target_text'], truncation=False)['input_ids']
        if len(source['input_ids']) > 512 or len(labels) > target_limit:
            raise ValueError(f"Token limit exceeded for {row['base_sample_id']}; no truncation allowed")
        if not labels or labels[-1] != tokenizer.eos_token_id:
            raise ValueError('Missing target EOS')
        features.append(dict(input_ids=source['input_ids'], attention_mask=source['attention_mask'], labels=labels))
        lengths['max_input_tokens'] = max(lengths['max_input_tokens'], len(source['input_ids']))
        lengths['max_target_tokens'] = max(lengths['max_target_tokens'], len(labels))
    return features, lengths


def generate(model, tokenizer, rows, mode, max_tokens, bf16):
    import torch
    was_training = model.training
    predictions = []
    started = time.perf_counter()
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.random.fork_rng(devices=[device.index or 0]), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            for row in rows:
                tokens = tokenizer(row['input_text'], return_tensors='pt', truncation=False)
                if tokens['input_ids'].shape[1] > 512:
                    raise ValueError('Input too long')
                inputs = {k: tokens[k].to(device) for k in ('input_ids', 'attention_mask')}
                ids = model.generate(**inputs, max_new_tokens=max_tokens, num_beams=1, do_sample=False)[0]
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                parsed = parse_output(raw, mode, row['grid'])
                predictions.append(dict(task='generation', base_sample_id=row['base_sample_id'], environment_id=row['environment_id'],
                    grid=row['grid'], mode=mode, input_text=row['input_text'], target_text=row['reference_action_text'],
                    target_trace_text=row['target_text'],
                    raw_output=raw, parsed=parsed, plan=parsed['plan'], parse_success=parsed['parse_success'],
                    candidate_plan=parsed['plan'] if parsed['parse_success'] else None,
                    action_exact_match=parsed['parse_success'] and parsed['plan'] == row['reference_actions'],
                    diagnostic_reason=diagnostic_execute(row['grid'], parsed['plan']) if parsed['parse_success'] else 'parse_failure',
                    generated_ids=ids.tolist(), hit_generation_limit=len(ids) - 1 >= max_tokens and ids[-1].item() != tokenizer.eos_token_id,
                    official_executor_label=None))
    finally:
        model.train(was_training)
    metrics = summarize(predictions, mode)
    metrics['inference_seconds'] = time.perf_counter() - started
    return metrics, predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('trace_only',), default='trace_only')
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=Path('data/processed/gen_compare_2000'))
    ap.add_argument('--dev-dir', type=Path, default=Path('data/processed/dev_expanded_v6/generation'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--evaluate-full-train', action='store_true', help='Optional extra inference over all 2000 train examples')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing overwrite: {args.output}')
    import os
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This experiment is fixed to one GPU')
    assets = verify_model(args.model_dir, 'flan-t5-base')
    started = time.perf_counter()
    train, dev, monitor, summary, dev_summary = load_data(args.data_dir, args.dev_dir)
    train, dev, monitor = ([make_example(row, args.mode) for row in subset] for subset in (train, dev, monitor))
    preparation_seconds = time.perf_counter() - started
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed
    if transformers.__version__ != '4.57.6' or torch.__version__.split('+')[0] != '2.9.1':
        raise RuntimeError('Use existing torch2.9.1 / transformers4.57.6 environment')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Submit the GPU job on a BF16-capable device')
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True)
    max_tokens = 512
    encoded_train, train_lengths = encode(train, tokenizer, max_tokens)
    encoded_dev, dev_lengths = encode(dev, tokenizer, max_tokens)
    # Check that generated training format survives tokenizer normalization and strict parsing.
    for row, feature in zip(train + dev, encoded_train + encoded_dev):
        decoded = tokenizer.decode(feature['labels'], skip_special_tokens=True)
        parsed = parse_output(decoded, args.mode, row['grid'])
        if not parsed['parse_success'] or parsed['plan'] != row['reference_actions']:
            raise ValueError('Reference format failed tokenizer roundtrip')
        if not all(parsed[k] is True for k in (
                'trace_parse_success', 'trace_transition_consistent',
                'trace_start_matches', 'trace_end_matches_goal')):
            raise ValueError('Reference trace failed tokenizer roundtrip')
    args.output.mkdir(parents=True)
    for name, rows in (('train', train), ('dev', dev)):
        write_rows(args.output / f'transformed_{name}.jsonl', rows)
    write_json(args.output / 'dataset_ids.json', {name: [dict(base_sample_id=r['base_sample_id'], environment_id=r['environment_id']) for r in rows]
                                               for name, rows in (('train', train), ('dev', dev))})
    training_args = TrainingArguments(output_dir=str(args.output / 'trainer'), num_train_epochs=3,
        per_device_train_batch_size=8, gradient_accumulation_steps=1, per_device_eval_batch_size=8,
        learning_rate=3e-4, lr_scheduler_type='constant', warmup_ratio=0, weight_decay=0.01,
        optim='adamw_torch', bf16=True, seed=42, data_seed=42, logging_steps=1,
        eval_strategy='epoch', save_strategy='no', report_to='none')
    report = dict(status='initialized', task='generation_trace_only_experiment', mode=args.mode, model=assets['repo_id'],
        model_assets=assets, seed=42, train_count=2000, dev_count=100, train_monitor_count=200,
        learning_rate=3e-4, batch_size=8, gradient_accumulation_steps=1, scheduler='constant', warmup_ratio=0,
        optimizer='adamw_torch', weight_decay=0.01, epochs_limit=3, bf16=True,
        preparation=summary, expanded_dev=dev_summary, cpu_representation_seconds=preparation_seconds,
        train_lengths=train_lengths, dev_lengths=dev_lengths,
        transformed_train_sha256=digest(args.output / 'transformed_train.jsonl'),
        transformed_dev_sha256=digest(args.output / 'transformed_dev.jsonl'),
        code_sha256={n: digest(Path(__file__).parent / n) for n in ('train_trace.py', 'state_format.py')},
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
        decoding=dict(num_beams=1, do_sample=False, max_new_tokens=max_tokens),
        checkpoint_selection='Dev action exact match, then lower Dev loss. Proxy only; official A success pending.',
        evaluation_note='Actions extracted only from the single Trace output. Diagnostic movement check is local; official A evaluation pending. No executor selection, correction, or true intermediate states at inference.',
        token_budget_note='Max 512 new tokens retained from v7 grid_state for a controlled comparison.')
    write_json(args.output / 'run_report.json', report)

    class Monitor(TrainerCallback):
        def __init__(self):
            self.history, self.best_key, self.best_metrics = [], None, None
            self.best_checkpoint = None

        def on_log(self, args_, state, control, logs=None, **kwargs):
            for key in ('loss', 'grad_norm', 'eval_loss'):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f'Nonfinite {key}')

        def on_evaluate(self, args_, state, control, model=None, metrics=None, **kwargs):
            folder = args.output / f'epoch_{round(state.epoch):02d}'
            folder.mkdir()
            values = dict(epoch=state.epoch, step=state.global_step, dev_loss=float(metrics['eval_loss']))
            for name, rows in (('train_monitor', monitor), ('dev', dev)):
                values[name], predictions = generate(model, tokenizer, rows, args.mode, max_tokens, True)
                write_rows(folder / f'{name}_predictions.jsonl', predictions)
            self.history.append(values)
            write_json(args.output / 'epoch_metrics.json', self.history)
            # Retain all three epochs so A can select by official Dev success later.
            epoch_checkpoint = folder / 'checkpoint'
            model.save_pretrained(epoch_checkpoint)
            tokenizer.save_pretrained(epoch_checkpoint)
            key = (values['dev']['action_exact_match_rate'], -values['dev_loss'])
            if self.best_key is None or key > self.best_key:
                self.best_key, self.best_metrics = key, values
                self.best_checkpoint = str(epoch_checkpoint)
            print(json.dumps({'epoch': state.epoch, 'dev': values['dev']}, ensure_ascii=False), flush=True)

    callback = Monitor()
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded_train, eval_dataset=encoded_dev,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), processing_class=tokenizer, callbacks=[callback])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    report.update(training_and_monitor_seconds=time.perf_counter() - started,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(), completed_steps=trainer.state.global_step,
                  completed_epochs=trainer.state.epoch, best_metrics=callback.best_metrics)
    write_json(args.output / 'train_log.json', dict(metrics=result.metrics, history=trainer.state.log_history))
    _, before = generate(model, tokenizer, dev[:1], args.mode, max_tokens, True)
    final = args.output / 'checkpoint-final'
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    del trainer, model
    torch.cuda.empty_cache()
    model = AutoModelForSeq2SeqLM.from_pretrained(final, local_files_only=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(final, local_files_only=True)
    _, after = generate(model, tokenizer, dev[:1], args.mode, max_tokens, True)
    roundtrip = dict(sample_id=dev[0]['base_sample_id'], generation_ids_equal=before[0]['generated_ids'] == after[0]['generated_ids'])
    write_json(args.output / 'save_reload_check.json', roundtrip)
    if not roundtrip['generation_ids_equal']:
        raise RuntimeError('Final checkpoint generation save/reload mismatch')
    report['final_metrics'] = {}
    for name, rows in (('train_monitor', monitor), ('dev', dev)):
        report['final_metrics'][name], predictions = generate(model, tokenizer, rows, args.mode, max_tokens, True)
        write_rows(args.output / f'{name}_predictions.jsonl', predictions)
    if args.evaluate_full_train:
        report['final_metrics']['train'], predictions = generate(model, tokenizer, train, args.mode, max_tokens, True)
        write_rows(args.output / 'train_predictions.jsonl', predictions)
    else:
        report['full_train_evaluation'] = 'Not requested; fixed original monitor200 evaluated instead.'
    report.update(status='complete', checkpoint=str(final), final_predictions_checkpoint='checkpoint-final',
                  dev_selected_checkpoint=callback.best_checkpoint, save_reload_check=roundtrip,
                  official_executor_evaluation_pending=True)
    write_json(args.output / 'run_report.json', report)
    print(f'Completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()
