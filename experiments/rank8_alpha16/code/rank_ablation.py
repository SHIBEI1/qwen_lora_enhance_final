#!/usr/bin/env python3
"""Run the approved rank-8/alpha-16 comparison using the frozen original run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path('/mnt/Disk1/kl/qwen_diffusion_enhance')
PYTHON = Path('/home/liu/miniforge3/envs/myenv/bin/python')
ACCELERATE = PYTHON.with_name('accelerate')
EPOCHS = (3, 6)
SEED = 3407


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def weight_info(path):
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        data = json.loads(handle.read(length))
    tensors = {key: value for key, value in data.items() if key.endswith(('.lora_down.weight', '.lora_up.weight'))}
    return {
        'path': str(path), 'sha256': sha(path), 'metadata': data.get('__metadata__', {}),
        'trainable_parameters': sum(math.prod(value['shape']) for value in tensors.values()),
        'modules': sorted(key.removesuffix('.lora_down.weight') for key in tensors if key.endswith('.lora_down.weight')),
        'ranks': sorted(set(value['shape'][0] for key, value in tensors.items() if key.endswith('.lora_down.weight'))),
    }


def pixel_hash(path):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB')
        digest = hashlib.sha256(struct.pack('<II', *image.size))
        digest.update(image.tobytes())
        return digest.hexdigest()


def paths(exp):
    exp = exp.resolve()
    if exp != ROOT / 'experiments/rank8_alpha16':
        raise RuntimeError(f'Unexpected experiment directory: {exp}')
    return exp


def status(exp, phase, **values):
    path = exp / 'status.json'
    prior = read_json(path) if path.exists() else {}
    prior.update(phase=phase, updated_at=stamp(), **values)
    write_json(path, prior)


def baseline_path(epoch):
    name = f'qwen_image_enhance_lora-{epoch:06d}.safetensors' if epoch < 6 else 'qwen_image_enhance_lora.safetensors'
    return ROOT / 'outputs/lora' / name


def rank8_path(exp, epoch):
    name = f'rank8_alpha16-{epoch:06d}.safetensors' if epoch < 6 else 'rank8_alpha16.safetensors'
    return exp / 'weights' / name


def environment():
    env = dict(os.environ)
    env.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', QDE_LOG_ACTIVE='1',
               PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'third_party/musubi-tuner/src'))
    return env


def train_command(exp):
    model = ROOT / 'models/hf/Qwen-Image-Edit-2511'
    return [str(ACCELERATE), 'launch', '--config_file', str(ROOT / 'configs/accelerate_single_gpu.yaml'),
            '--num_cpu_threads_per_process', '1', '--mixed_precision', 'bf16',
            str(ROOT / 'third_party/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py'),
            '--dit', str(model / 'transformer/diffusion_pytorch_model-00001-of-00005.safetensors'),
            '--vae', str(model / 'vae/diffusion_pytorch_model.safetensors'),
            '--text_encoder', str(model / 'text_encoder/model-00001-of-00004.safetensors'),
            '--dataset_config', str(exp / 'configs/train.toml'), '--model_version', 'edit-2511', '--sdpa',
            '--mixed_precision', 'bf16', '--timestep_sampling', 'shift', '--weighting_scheme', 'none',
            '--discrete_flow_shift', '2.2', '--optimizer_type', 'adamw8bit', '--learning_rate', '5e-5',
            '--gradient_checkpointing', '--blocks_to_swap', '16', '--fp8_base', '--fp8_scaled',
            '--max_data_loader_n_workers', '2', '--persistent_data_loader_workers',
            '--gradient_accumulation_steps', '1', '--network_module', 'networks.lora_qwen_image',
            '--network_dim', '8', '--network_alpha', '16', '--network_dropout', '0.05',
            '--max_train_epochs', '6', '--save_every_n_epochs', '1', '--seed', str(SEED),
            '--output_dir', str(exp / 'weights'), '--output_name', 'rank8_alpha16']


def prepare(exp):
    exp = paths(exp)
    if (exp / 'protocol.json').exists():
        verify_inputs(exp)
        print('Using the already prepared, frozen protocol.', flush=True)
        return
    for folder in ['configs', 'logs', 'weights', 'results', 'reports', 'source_images/input', 'source_images/target', 'snapshots']:
        (exp / folder).mkdir(parents=True, exist_ok=True)
    status(exp, 'preparing', pid=os.getpid(), started_at=stamp())
    train = records(ROOT / 'dataset/manifests/train.jsonl')
    val = records(ROOT / 'dataset/manifests/val.jsonl')
    original_selection = records(ROOT / 'checkpoint_selection/selection_manifest.jsonl')
    if len(train) != 640 or Counter(row['task'] for row in train) != Counter(deblur=320, lowlight=320):
        raise RuntimeError('The original training split no longer has 320+320 paired images.')
    baseline = {str(epoch): weight_info(baseline_path(epoch)) for epoch in EPOCHS}
    for epoch, info in baseline.items():
        m = info['metadata']
        expected = {'ss_network_dim': '16', 'ss_network_alpha': '32.0', 'ss_seed': '3407',
                    'ss_learning_rate': '5e-05', 'ss_num_epochs': '6', 'ss_num_train_items': '640',
                    'ss_network_dropout': '0.05', 'ss_gradient_accumulation_steps': '1',
                    'ss_steps': str(int(epoch) * 640)}
        for key, value in expected.items():
            if m.get(key) != value:
                raise RuntimeError(f'Unexpected baseline setting {epoch}: {key}={m.get(key)}')
        if info['ranks'] != [16] or len(info['modules']) != 840 or info['trainable_parameters'] != 147456000:
            raise RuntimeError('Baseline LoRA injection scope changed.')
    print('Decoding original train/val images to check exact RGB content overlap.', flush=True)
    training_hashes = set()
    unique_pairs = set()
    training_scenes = {row['source_scene'] for row in train if row['task'] == 'deblur'}
    for index, row in enumerate(train, 1):
        pair = tuple(pixel_hash(ROOT / 'dataset' / row[key]) for key in ('input', 'target'))
        training_hashes.update(pair)
        unique_pairs.add(pair)
        if index % 160 == 0:
            print(f'Pixel audit: train {index}/640', flush=True)
    eligible, audit = {}, []
    for row in val:
        pair = tuple(pixel_hash(ROOT / 'dataset' / row[key]) for key in ('input', 'target'))
        reason = []
        if any(value in training_hashes for value in pair):
            reason.append('input_or_target_exact_RGB_overlap_with_training')
        if row['task'] == 'deblur' and row['source_scene'] in training_scenes:
            reason.append('GoPro_scene_overlap_with_training')
        audit.append({'id': row['id'], 'task': row['task'], 'input_pixel_sha256': pair[0],
                      'target_pixel_sha256': pair[1], 'eligible': not reason, 'excluded_reasons': reason})
        if not reason:
            eligible[row['id']] = (row, pair)
    selected, used_hashes, replacements = [], set(), []
    excluded_pilot = set(read_json(ROOT / 'checkpoint_selection/protocol.json').get('pilot_validation_ids_excluded', []))
    for original in original_selection:
        entry = eligible.get(original['id'])
        if entry is None or any(value in used_hashes for value in entry[1]):
            pool = [(row, pair) for row, pair in eligible.values()
                    if row['task'] == original['task'] and row['id'] not in excluded_pilot
                    and row['id'] not in {item['id'] for item in original_selection}
                    and not any(value in used_hashes for value in pair)]
            # Match input brightness, without considering generated-output metrics.
            pool.sort(key=lambda item: (abs(float(item[0]['input_meta']['luma']) - float(original['input_meta']['luma'])), item[0]['id']))
            if not pool:
                raise RuntimeError('No clean replacement validation image is available.')
            entry = pool[0]
            replacements.append({'removed_id': original['id'], 'replacement_id': entry[0]['id'],
                                 'reason': 'exclude RGB content overlap; replace by closest input brightness'})
        row, pair = entry
        row = dict(row, selection_index=len(selected) + 1, inference_seed=SEED)
        selected.append(row)
        used_hashes.update(pair)
    if Counter(row['task'] for row in selected) != Counter(deblur=3, lowlight=3) or len(selected) != 6:
        raise RuntimeError('Expected exactly six distinct, training-content-disjoint paired samples.')
    manifest = exp / 'configs/clean_val.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    source_hashes = {}
    for row in selected:
        for key in ('input', 'target'):
            source = ROOT / 'dataset' / row[key]
            destination = exp / 'source_images' / key / row['task'] / f"{row['id']}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hashes[str(source)] = sha(source)
    source_files = ['dataset/manifests/train.jsonl', 'dataset/manifests/val.jsonl',
                    'artifacts/dataset/train.jsonl', 'artifacts/dataset/train.toml',
                    'configs/accelerate_single_gpu.yaml', 'configs/rtx3090_24gb.env',
                    'scripts/train_lora.sh', 'scripts/enhance.py', 'scripts/evaluate.py',
                    'qwen_enhance/constants.py', 'qwen_enhance/runtime_logging.py',
                    'third_party/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py',
                    'third_party/musubi-tuner/src/musubi_tuner/networks/lora.py',
                    'third_party/musubi-tuner/src/musubi_tuner/networks/lora_qwen_image.py']
    for name in source_files:
        source = ROOT / name
        destination = exp / 'snapshots' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(source)] = sha(source)
    cache = ROOT / 'artifacts/cache/train_512x512_ctrl512'
    cached = sorted(cache.glob('*.safetensors'))
    if len(cached) != 1280:
        raise RuntimeError(f'Expected original 1280 cached feature files, found {len(cached)}.')
    cache_inventory = [{'name': file.name, 'size': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns} for file in cached]
    write_json(exp / 'snapshots/cache_inventory.json', cache_inventory)
    # Only copy the existing config: cache paths and train image order stay identical.
    shutil.copy2(ROOT / 'artifacts/dataset/train.toml', exp / 'configs/train.toml')
    write_json(exp / 'reports/content_overlap_audit.json', {'train_pairs': len(train), 'unique_decoded_train_pairs': len(unique_pairs),
               'val_audit': audit, 'selected_ids': [row['id'] for row in selected], 'replacements': replacements,
               'limitation': 'Exact decoded RGB audit and GoPro scene isolation; this does not certify absence of lowlight near-duplicates.'})
    protocol = {
        'created_at': stamp(), 'project_root': str(ROOT), 'experiment_root': str(exp),
        'purpose': 'rank8/alpha16 versus rank16/alpha32 with all other training settings fixed',
        'training': {'train_pairs': 640, 'epochs': 6, 'steps_per_epoch': 640, 'max_steps': 3840,
                     'seed': SEED, 'rank': 8, 'alpha': 16, 'alpha_over_rank': 2.0, 'dropout': 0.05,
                     'learning_rate': 5e-5, 'batch_size': 1, 'gradient_accumulation': 1,
                     'resolution': [512, 512], 'optimizer': 'adamw8bit', 'mixed_precision': 'bf16',
                     'fp8_base': True, 'fp8_scaled': True, 'blocks_to_swap': 16,
                     'initialization': 'fresh LoRA from frozen original Qwen base; no rank16 adapter resume',
                     'target_scope': 'the same 840 matrices as the baseline; no injection-scope change'},
        'evaluation': {'split': 'val', 'count': 6, 'task_counts': {'deblur': 3, 'lowlight': 3},
                       'selected_ids': [row['id'] for row in selected], 'replacements': replacements,
                       'primary_epoch': 3, 'secondary_epoch': 6, 'steps': 25, 'seed': SEED,
                       'guidance_scale': 4.0, 'blocks_to_swap': 16, 'max_pixels': 1048576,
                       'lora_multiplier': 1.0, 'metric': 'full RGB PSNR/SSIM at original output size, same original evaluator',
                       'rule': 'paired matched-epoch comparisons; report both tasks and metrics, no best-checkpoint search'},
        'baseline_weights': baseline, 'source_sha256': source_hashes,
        'limitations': ['one training seed', 'six validation examples, not a final test benchmark',
                       'lowlight near-duplicate scenes not certified', 'rank-dependent initialization may change RNG consumption'],
    }
    write_json(exp / 'protocol.json', protocol)
    write_json(exp / 'configs/train_command.json', {'argv': train_command(exp)})
    (exp / 'README.md').write_text(
        '# Rank 8 / alpha 16 comparison\n\n'
        'Run from the original development project using the existing myenv environment.\n'
        'Training reuses exactly the original 640 pairs and frozen feature caches.\n'
        'The primary comparison is epoch 3 / step 1920; epoch 6 / step 3840 is secondary.\n'
        'Both ranks share three lowlight and three deblur images after decoded-RGB overlap checks.\n\n'
        '- `protocol.json`: frozen settings, source hashes, and baseline metadata.\n'
        '- `configs/`: exact training command, frozen train config and clean evaluation manifest.\n'
        '- `weights/`: this rank-8 run only.\n'
        '- `source_images/`: physically copied input and target pairs.\n'
        '- `results/rank*_epoch*/`: generated images, per-image CSV and aggregate metrics.\n'
        '- `reports/comparison.json` and `comparison.md`: paired comparison and limitations.\n'
        '- `logs/`: timestamped fixed-name logs; an intentional rerun overwrites only its own log.\n'
        '- `status.json`: pipeline phase, step count, PID and errors.\n\n'
        'Inspect progress: `python code/rank_ablation.py status` from this experiment directory.\n'
        'The published final project and selected rank-16 adapter are not replaced by this experiment.\n', encoding='utf-8')
    status(exp, 'prepared', selected_ids=[row['id'] for row in selected], replacements=replacements)
    print(json.dumps({'phase': 'prepared', 'selected_ids': protocol['evaluation']['selected_ids'],
                      'replacements': replacements, 'eligible_val_pairs': len(eligible),
                      'unique_train_pairs': len(unique_pairs)}, ensure_ascii=False, indent=2), flush=True)


def verify_inputs(exp):
    protocol = read_json(exp / 'protocol.json')
    for filename, expected in protocol['source_sha256'].items():
        if sha(filename) != expected:
            raise RuntimeError(f'Frozen source changed: {filename}')
    for info in protocol['baseline_weights'].values():
        if sha(info['path']) != info['sha256']:
            raise RuntimeError(f'Baseline adapter changed: {info["path"]}')
    for item in read_json(exp / 'snapshots/cache_inventory.json'):
        file = ROOT / 'artifacts/cache/train_512x512_ctrl512' / item['name']
        stat = file.stat()
        if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
            raise RuntimeError(f'Frozen feature cache changed: {file}')


def run_command(exp, argv, log_name, phase):
    log_path = exp / 'logs' / f'{log_name}.log'
    write_json(exp / 'configs' / f'{log_name}_command.json', {'argv': argv, 'cwd': str(ROOT)})
    print(f'[{stamp()}] Starting {phase}; log={log_path}', flush=True)
    start = time.monotonic()
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        status(exp, phase, child_pid=process.pid, phase_started_at=stamp(), phase_log=str(log_path))
        log.write(f'[{stamp()}] COMMAND {json.dumps(argv)}\n')
        assert process.stdout is not None
        previous_step, previous_update = 0, time.monotonic()
        for line in process.stdout:
            rendered = f'[{stamp()}] {line.rstrip()}\n'
            log.write(rendered)
            # The outer timestamp_tee adds the terminal/pipeline timestamp;
            # this stage's own file already receives its stamped line above.
            sys.stdout.write(line)
            sys.stdout.flush()
            if phase == 'training':
                match = re.search(r'(\d+)/3840.*?avr_loss=([0-9.eE+-]+)', line)
                if match:
                    step = int(match.group(1))
                    if step >= previous_step + 20 or time.monotonic() - previous_update > 30:
                        status(exp, phase, global_step=step, max_steps=3840,
                               training_avr_loss=float(match.group(2)), training_elapsed_seconds=round(time.monotonic()-start, 1))
                        previous_step, previous_update = step, time.monotonic()
        code = process.wait()
        log.write(f'[{stamp()}] EXIT {code}; seconds={time.monotonic()-start:.1f}\n')
        if code:
            raise subprocess.CalledProcessError(code, argv)
    return time.monotonic() - start


def validate_rank8(exp):
    protocol = read_json(exp / 'protocol.json')
    baseline = protocol['baseline_weights']['3']
    infos = {}
    for epoch in EPOCHS:
        info = weight_info(rank8_path(exp, epoch))
        m = info['metadata']
        if info['ranks'] != [8] or info['modules'] != baseline['modules'] or info['trainable_parameters'] != 73728000:
            raise RuntimeError('Rank-8 adapter scope/parameter count does not match the controlled comparison.')
        for key, value in [('ss_network_dim','8'), ('ss_network_alpha','16.0'), ('ss_steps',str(epoch*640)),
                           ('ss_num_train_items','640'), ('ss_num_epochs','6'), ('ss_learning_rate','5e-05'),
                           ('ss_seed','3407'), ('ss_network_dropout','0.05')]:
            if m.get(key) != value:
                raise RuntimeError(f'Unexpected rank-8 setting: {key}={m.get(key)}')
        infos[str(epoch)] = info
    write_json(exp / 'reports/rank8_weight_audit.json', infos)
    return infos


def reuse_baseline(exp, epoch):
    from PIL import Image
    protocol = read_json(exp / 'protocol.json')
    old_label = f'epoch_{epoch:02d}' if epoch < 6 else 'epoch_06_final'
    old_root = ROOT / 'checkpoint_selection/results' / old_label
    destination = exp / 'results' / f'rank16_epoch{epoch:02d}'
    expected_lora = baseline_path(epoch).resolve()
    reused = []
    sys.path.insert(0, str(ROOT))
    from qwen_enhance.constants import FIXED_ENHANCEMENT_PROMPT
    for row in records(exp / 'configs/clean_val.jsonl'):
        source = old_root / row['task'] / f"{row['id']}.png"
        metadata_path = source.with_suffix('.png.json')
        if not source.exists() or not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if Path(metadata['lora']).resolve() != expected_lora or metadata['prompt'] != FIXED_ENHANCEMENT_PROMPT:
            raise RuntimeError(f'Old baseline provenance mismatch: {source}')
        if Path(metadata['input']).resolve() != (ROOT / 'dataset' / row['input']).resolve():
            raise RuntimeError(f'Old baseline input mismatch: {source}')
        for key in ['steps', 'seed', 'guidance_scale', 'blocks_to_swap']:
            if metadata[key] != protocol['evaluation'][key]:
                raise RuntimeError(f'Old baseline inference setting differs: {key}')
        with Image.open(source) as image:
            if image.size != (row['target_meta']['width'], row['target_meta']['height']):
                raise RuntimeError(f'Old baseline image dimensions differ: {source}')
            image.verify()
        target = destination / row['task'] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        metadata.update(output=str(target), reused_from=str(source), reused_source_sha256=sha(source))
        write_json(target.with_suffix('.png.json'), metadata)
        reused.append({'id': row['id'], 'source': str(source), 'prediction_sha256': sha(source)})
    write_json(destination / 'reuse_provenance.json', reused)
    print(f'Baseline epoch {epoch}: reusing {len(reused)} verified old predictions; generating the rest.', flush=True)


def evaluate(exp, rank, epoch):
    label = f'rank{rank}_epoch{epoch:02d}'
    adapter = baseline_path(epoch) if rank == 16 else rank8_path(exp, epoch)
    if rank == 16:
        reuse_baseline(exp, epoch)
    command = [str(PYTHON), str(ROOT / 'scripts/evaluate.py'), '--project-root', str(ROOT),
               '--lora', str(adapter), '--manifest', str(exp / 'configs/clean_val.jsonl'),
               '--label', label, '--output-dir', str(exp / 'results' / label),
               '--steps', '25', '--seed', str(SEED), '--skip-existing',
               '--log-name', label, '--log-dir', str(exp / 'logs')]
    elapsed = run_command(exp, command, label, f'evaluating_{label}')
    result = read_json(exp / 'results' / label / 'metrics.json')
    if result['count'] != 6 or {task: value['count'] for task,value in result['by_task'].items()} != {'deblur':3,'lowlight':3}:
        raise RuntimeError('Evaluation output does not match the six-image protocol.')
    return elapsed


def image_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def loss_boundaries(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        match = re.search(r'(\d+)/3840.*?avr_loss=([0-9.eE+-]+)', line)
        if match and int(match.group(1)) in [640,1280,1920,2560,3200,3840]:
            values[str(int(match.group(1))//640)] = float(match.group(2))
    return values


def compare(exp):
    from PIL import Image
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    protocol = read_json(exp / 'protocol.json')
    pairs = []
    for epoch in EPOCHS:
        summaries = {rank: read_json(exp / 'results' / f'rank{rank}_epoch{epoch:02d}' / 'metrics.json') for rank in (8,16)}
        rows = {rank: {row['id']: row for row in image_rows(exp / 'results' / f'rank{rank}_epoch{epoch:02d}' / 'per_image_metrics.csv')} for rank in (8,16)}
        if set(rows[8]) != set(rows[16]) or set(rows[8]) != set(protocol['evaluation']['selected_ids']):
            raise RuntimeError('Ranks were not evaluated on exactly the same images.')
        deltas = []
        for item in protocol['evaluation']['selected_ids']:
            a,b = rows[8][item],rows[16][item]
            if a['input'] != b['input'] or a['target'] != b['target']:
                raise RuntimeError('Paired metrics refer to different input/target files.')
            deltas.append({'id': item, 'task': a['task'], 'rank8_psnr': float(a['psnr']), 'rank16_psnr': float(b['psnr']),
                           'delta_psnr': float(a['psnr'])-float(b['psnr']), 'rank8_ssim': float(a['ssim']),
                           'rank16_ssim': float(b['ssim']), 'delta_ssim': float(a['ssim'])-float(b['ssim'])})
        group = {}
        for task in ['deblur','lowlight']:
            group[task] = {key: summaries[8]['by_task'][task][key]-summaries[16]['by_task'][task][key] for key in ['psnr','ssim']}
        group['macro'] = {key: mean(group[task][key] for task in ['deblur','lowlight']) for key in ['psnr','ssim']}
        pairs.append({'epoch': epoch, 'step': epoch*640, 'rank8': summaries[8], 'rank16': summaries[16],
                      'deltas_rank8_minus_rank16': group, 'per_image': deltas})
    input_metrics = []
    for row in records(exp / 'configs/clean_val.jsonl'):
        with Image.open(ROOT/'dataset'/row['input']) as image: image_in=np.asarray(image.convert('RGB'))
        with Image.open(ROOT/'dataset'/row['target']) as image: target=np.asarray(image.convert('RGB'))
        if image_in.shape != target.shape: raise RuntimeError('Input/target dimensions differ.')
        input_metrics.append({'id': row['id'], 'task':row['task'], 'psnr':float(peak_signal_noise_ratio(target,image_in,data_range=255)),
                              'ssim':float(structural_similarity(target,image_in,channel_axis=-1,data_range=255))})
    report = {'created_at':stamp(), 'protocol':protocol['evaluation'], 'training_settings':protocol['training'],
              'comparison':pairs, 'input_metrics':input_metrics,
              'epoch_boundary_training_avr_loss':{'rank8':loss_boundaries(exp/'logs/train_rank8.log'),
                                                   'rank16':loss_boundaries(ROOT/'logs/train.log')},
              'limitations':protocol['limitations'],
              'interpretation':'Use the epoch-3 paired comparison as the primary answer. Epoch 6 describes additional-training behavior. '
                               'Small-sample metric gains alone do not establish general superiority or diagnose overfitting.'}
    write_json(exp/'reports/comparison.json',report)
    table=['# Rank 8 / alpha 16 versus rank 16 / alpha 32', '',
           'Primary comparison: matched epoch 3 / step 1920. Secondary: epoch 6 / step 3840.', '',
           '| Epoch | Task | r8 PSNR | r16 PSNR | Delta PSNR | r8 SSIM | r16 SSIM | Delta SSIM |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pair in pairs:
        for task in ['deblur','lowlight']:
            a,b=pair['rank8']['by_task'][task],pair['rank16']['by_task'][task]
            table.append(f"| {pair['epoch']} | {task} | {a['psnr']:.4f} | {b['psnr']:.4f} | {a['psnr']-b['psnr']:+.4f} | {a['ssim']:.4f} | {b['ssim']:.4f} | {a['ssim']-b['ssim']:+.4f} |")
    table += ['', 'All settings other than LoRA rank and alpha were held fixed; both configurations have alpha/rank=2.',
              'Training data stayed unchanged. Evaluation excludes exact decoded RGB overlap with any training input/target.',
              'This is a six-image, single-seed validation experiment. Lowlight near-duplicate scenes are not certified absent.',
              'Original input metrics and per-image differences are retained in comparison.json; visually inspect the generated panels.',
              'No final release adapter is automatically replaced by this experiment.', '']
    (exp/'reports/comparison.md').write_text('\n'.join(table),encoding='utf-8')
    with (exp/'reports/paired_deltas.csv').open('w',encoding='utf-8',newline='') as handle:
        rows=[dict(epoch=pair['epoch'],**row) for pair in pairs for row in pair['per_image']]
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=records(exp/'configs/clean_val.jsonl')
    for epoch in EPOCHS:
        fig,axes=plt.subplots(6,4,figsize=(18,18),layout='constrained')
        for index,row in enumerate(source):
            files=[ROOT/'dataset'/row['input'],ROOT/'dataset'/row['target'],
                   exp/'results'/f'rank16_epoch{epoch:02d}'/row['task']/f"{row['id']}.png",
                   exp/'results'/f'rank8_epoch{epoch:02d}'/row['task']/f"{row['id']}.png"]
            for column,(file,title) in enumerate(zip(files,['Input','Target','Rank 16','Rank 8'])):
                with Image.open(file) as image: axes[index,column].imshow(np.asarray(image.convert('RGB')))
                axes[index,column].axis('off')
                axes[index,column].set_title(f"{title} | {row['id']}",fontsize=9)
        fig.suptitle(f'Matched epoch {epoch}; 25 inference steps; seed 3407',fontsize=16)
        fig.savefig(exp/'reports'/f'comparison_epoch{epoch:02d}.jpg',dpi=150)
        plt.close(fig)
    return report


def run(exp):
    import fcntl
    exp=paths(exp)
    prepare(exp)
    with (exp/'pipeline.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('The experiment pipeline is already running.')
        prior=read_json(exp/'status.json')
        if prior['phase']=='complete':
            print('This experiment is already complete. Read reports/comparison.json.',flush=True)
            return
        status(exp,'checking',pid=os.getpid(),started_at=stamp())
        try:
            verify_inputs(exp)
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
            if gpu: raise RuntimeError(f'The GPU is in use by another process: {gpu}')
            if not rank8_path(exp,6).exists():
                existing=list((exp/'weights').glob('*.safetensors'))
                if existing: raise RuntimeError('Partial rank8 checkpoints exist; do not silently overwrite or resume them.')
                train_seconds=run_command(exp,train_command(exp),'train_rank8','training')
                status(exp,'training_finished',global_step=3840,training_seconds=round(train_seconds,1))
            validate_rank8(exp)
            for epoch in EPOCHS:
                for rank in (16,8):
                    elapsed=evaluate(exp,rank,epoch)
                    status(exp,'evaluation_checkpoint_finished',last_label=f'rank{rank}_epoch{epoch:02d}',last_evaluation_seconds=round(elapsed,1))
            verify_inputs(exp)
            status(exp,'comparing')
            compare(exp)
            status(exp,'complete',completed_at=stamp(),report=str(exp/'reports/comparison.json'))
            print(f'[{stamp()}] Complete. Report: {exp}/reports/comparison.json',flush=True)
        except BaseException as exc:
            status(exp,'failed',error=repr(exc),failed_at=stamp())
            traceback.print_exc()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status','compare'])
    parser.add_argument('--experiment-dir',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();exp=paths(args.experiment_dir)
    if args.action=='prepare': prepare(exp)
    elif args.action=='run': run(exp)
    elif args.action=='compare': compare(exp)
    else:
        state=read_json(exp/'status.json')
        for key in ['pid','child_pid']:
            if state.get(key): state[key+'_alive']=(Path('/proc')/str(state[key])).exists()
        print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
