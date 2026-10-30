#!/usr/bin/env python3
"""Bounded CPU LoRA feasibility pilot on the official GSM8K train split."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time


GSM8K_TRAIN_SHA256 = '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'
GSM8K_TEST_SHA256 = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
PILOT_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def stop_requested(_signum, _frame):
    raise TimeoutError('LoRA pilot deadline reached')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_rows(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {'question', 'answer'}:
                raise ValueError(f'Invalid GSM8K training row {number}')
            if not all(isinstance(item[key], str) and item[key].strip()
                       for key in ('question', 'answer')):
                raise ValueError(f'Empty GSM8K training row {number}')
            yield number, item


def encode_sample(tokenizer, row, sequence_length):
    instruction = 'Solve the math problem. Show the calculation and final answer.'
    prompt = [{'role': 'system', 'content': instruction},
              {'role': 'user', 'content': row['question']}]
    full = prompt + [{'role': 'assistant', 'content': row['answer']}]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True,
                                               add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(full, tokenize=True,
                                             add_generation_prompt=False)
    if (len(full_ids) > sequence_length or len(full_ids) <= len(prompt_ids)
            or full_ids[:len(prompt_ids)] != prompt_ids):
        return None
    return full_ids, [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]


def heldout_loss(model, samples):
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for ids, labels in samples:
            loss = model(input_ids=torch.tensor([ids], dtype=torch.long),
                         labels=torch.tensor([labels], dtype=torch.long),
                         use_cache=False).loss
            losses.append(float(loss.float()))
    model.train()
    return sum(losses) / len(losses)


def main():
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--eval-data', type=Path)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    if (not 1 <= args.max_steps <= 32 or not 64 <= args.sequence_length <= 256 or
            not 1e-6 <= args.learning_rate <= 1e-4):
        parser.error('Pilot bounds are 1-32 steps, 64-256 tokens, and 1e-6 to 1e-4 LR')
    if not args.model_dir.is_dir() or not args.data.is_file():
        parser.error('Model directory and training data must exist')
    actual_hash = file_sha256(args.data)
    if actual_hash != GSM8K_TRAIN_SHA256:
        parser.error('Training data SHA-256 does not match official GSM8K train.jsonl')
    if args.eval_data and (not args.eval_data.is_file() or
                           file_sha256(args.eval_data) != GSM8K_TEST_SHA256):
        parser.error('Held-out data SHA-256 does not match official GSM8K test.jsonl')
    result = {'model_dir': str(args.model_dir), 'data_sha256': actual_hash,
              'adapter_dir': str(args.adapter_dir), 'max_steps': args.max_steps,
              'sequence_length': args.sequence_length, 'learning_rate': args.learning_rate,
              'losses': [],
              'selected_rows': [], 'status': 'running'}
    started = time.monotonic()
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
        if not tokenizer.chat_template:
            tokenizer.chat_template = PILOT_CHAT_TEMPLATE
            result['chat_template'] = 'Qwen role delimiters matching the GGUF template'
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map='cpu', attn_implementation='eager', trust_remote_code=False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.0, bias='none', task_type=TaskType.CAUSAL_LM))
        model.enable_input_require_grads()
        model.train()
        if args.eval_data:
            samples = []
            result['selected_eval_rows'] = []
            for number, row in training_rows(args.eval_data):
                encoded = encode_sample(tokenizer, row, args.sequence_length)
                if encoded is not None:
                    samples.append(encoded)
                    result['selected_eval_rows'].append(number)
                if len(samples) == 4:
                    break
            if len(samples) != 4:
                raise ValueError('Not enough held-out rows fit the sequence limit')
            result['heldout_loss_before'] = heldout_loss(model, samples)
        result['trainable_parameters'] = sum(p.numel() for p in model.parameters()
                                             if p.requires_grad)
        print(json.dumps({'loaded': True, 'trainable_parameters': result['trainable_parameters']}),
              flush=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.learning_rate)
        for number, row in training_rows(args.data):
            encoded = encode_sample(tokenizer, row, args.sequence_length)
            if encoded is None:
                continue
            ids, labels = encoded
            input_ids = torch.tensor([ids], dtype=torch.long)
            target_ids = torch.tensor([labels], dtype=torch.long)
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=target_ids,
                         use_cache=False).loss
            loss.backward()
            optimizer.step()
            result['losses'].append(float(loss.detach().float()))
            result['selected_rows'].append(number)
            print(json.dumps({'step': len(result['losses']), 'row': number,
                              'loss': result['losses'][-1],
                              'seconds': round(time.monotonic() - step_started, 2)}),
                  flush=True)
            if len(result['losses']) >= args.max_steps:
                break
        if len(result['losses']) != args.max_steps:
            raise ValueError('Not enough training rows fit the sequence limit')
        if args.eval_data:
            result['heldout_loss_after'] = heldout_loss(model, samples)
        args.adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.adapter_dir)
        tokenizer.save_pretrained(args.adapter_dir)
        result['status'] = 'completed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['wall_seconds'] = round(time.monotonic() - started, 2)
        result['peak_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'],
                      'peak_rss_mib': result['peak_rss_mib']}), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
