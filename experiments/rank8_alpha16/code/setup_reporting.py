#!/usr/bin/env python3
"""Add the plotting dependency without changing validated training libraries."""
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

exp = Path(__file__).resolve().parents[1]
core = ['numpy', 'pillow', 'scikit-image', 'torch', 'transformers', 'accelerate', 'safetensors', 'bitsandbytes']
before = {name: version(name) for name in core}
constraints = exp / 'configs/core_package_constraints.txt'
constraints.write_text(''.join(f'{name}=={value}\n' for name, value in before.items()), encoding='utf-8')
command = [sys.executable, '-m', 'pip', 'install', '--constraint', str(constraints), 'matplotlib==3.10.7']
with (exp / 'logs/setup_reporting.log').open('w', encoding='utf-8', buffering=1) as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        message = f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {line}'
        log.write(message)
        print(message, end='', flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
after = {name: version(name) for name in core}
if before != after:
    raise RuntimeError(f'Validated core package versions changed: {before} -> {after}')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figure = plt.figure()
plt.close(figure)
audit = {'before': before, 'after': after, 'core_versions_unchanged': True,
         'added_matplotlib': version('matplotlib'), 'plotting_import_verified': True}
(exp / 'reports/reporting_environment.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
print(json.dumps(audit, indent=2), flush=True)
