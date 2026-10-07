#!/usr/bin/env python3
"""Record an explicitly requested pause after the experiment has stopped."""
import json
import os
import re
from datetime import datetime
from pathlib import Path


EXP = Path('/mnt/Disk1/kl/qwen_diffusion_enhance/experiments/rank8_alpha16')


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    if Path(__file__).resolve().parent != EXP / 'code':
        raise RuntimeError('This helper is only for the existing remote experiment.')
    state_path = EXP / 'status.json'
    prior = json.loads(state_path.read_text(encoding='utf-8'))
    if prior['phase'] == 'complete':
        raise RuntimeError('Completed experiments must not be marked paused.')
    active = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            argv = (item / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (str(EXP / 'code/run_pipeline.sh') in argv
                or (str(EXP / 'code/rank_ablation.py') in argv and 'run' in argv)
                or ('--output_dir' in argv and str(EXP / 'weights') in argv)
                or ('--output-dir' in argv and any(arg.startswith(str(EXP / 'results') + '/') for arg in argv))):
            active.append(int(item.name))
    if active:
        raise RuntimeError(f'Experiment processes still active; refusing to mark paused: {active}')
    weights = sorted(EXP.joinpath('weights').glob('*.safetensors'))
    log = EXP / 'logs/train_rank8.log'
    progress = re.findall(r'(\d+)/3840.*?avr_loss=([0-9.eE+-]+)', log.read_text(encoding='utf-8', errors='replace'))
    step, loss = max(((int(step), float(loss)) for step, loss in progress), default=(prior.get('global_step', 0), prior.get('training_avr_loss')))
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    resume_mode = 'restart_fresh_from_original_frozen_configuration' if not weights else 'inspect_saved_checkpoints_before_restart'
    record = {
        'paused_at': now,
        'reason': 'User requested stopping this experiment while developing another project.',
        'last_logged_step': step,
        'max_steps': 3840,
        'last_logged_avr_loss': loss,
        'saved_weight_files': [str(path) for path in weights],
        'full_training_state_saved': False,
        'resume_mode': resume_mode,
        'automatic_resume': False,
        'automation_id': 'qwen-rank8',
        'automation_status': 'PAUSED',
        'prior_state': prior,
    }
    write_json(EXP / 'reports/pause_record.json', record)
    updated = dict(prior)
    interruption = {key: updated.pop(key) for key in ('error', 'failed_at') if key in updated}
    updated.update(phase='paused', updated_at=now, paused_at=now,
                   pause_reason=record['reason'], global_step=step,
                   training_avr_loss=loss, automatic_resume=False,
                   resume_mode=resume_mode, pause_record=str(EXP / 'reports/pause_record.json'))
    if interruption:
        updated['intentional_stop_details'] = interruption
    write_json(state_path, updated)
    message = (f'[{now}] USER-REQUESTED PAUSE; processes stopped; last logged step={step}/3840; '
               f'saved weights={len(weights)}; resume mode={resume_mode}; automatic resume disabled.\n')
    for path in (EXP / 'logs/pause.log', EXP / 'logs/pipeline.log'):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(message)
    print(json.dumps({key: value for key, value in record.items() if key != 'prior_state'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
