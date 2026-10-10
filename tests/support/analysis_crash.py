"""Real host/guardian launch checkpoints; production ownership is unchanged."""

HOST = r"""
import asyncio, json, subprocess, sys
from pathlib import Path
from daita import Agent
from daita.adapters.analytical_workspace.native import NativePythonWorker
from daita.storage.sql import SQLStateStore
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

root, boundary = Path(sys.argv[1]), sys.argv[2]
metadata, checkpoint, release = (root / name for name in ('metadata.json', 'checkpoint', 'release'))
guardian = Path(__import__('daita.adapters.analytical_workspace.guardian', fromlist=['x']).__file__)
shim = root / 'guardian_checkpoint.py'
shim.write_text('''import json, os, runpy, subprocess, sys, time
from pathlib import Path
checkpoint, release = Path(%r), Path(%r)
boundary = %r
def stop():
    checkpoint.write_text('ready')
    deadline = time.monotonic() + 30
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('test checkpoint expired')
        time.sleep(.01)
original = subprocess.Popen
def spawn(*args, **kwargs):
    if boundary == 'before_worker': stop()
    worker = original(*args, **kwargs)
    if boundary == 'after_worker': stop()
    return worker
subprocess.Popen = spawn
if boundary == 'guardian_inherited': stop()
runpy.run_path(%r, run_name='__main__')
''' % (str(checkpoint), str(release), boundary, str(guardian)))

original_popen = subprocess.Popen
def launch(args, *other, **kwargs):
    if str(guardian) in args:
        if boundary == 'handoff':
            checkpoint.write_text('ready')
            while not release.exists():
                __import__('time').sleep(.01)
        args = [str(shim) if item == str(guardian) else item for item in args]
    return original_popen(args, *other, **kwargs)
subprocess.Popen = launch
original_event = NativePythonWorker._guardian_event
async def event(worker, deadline):
    value = await original_event(worker, deadline)
    if boundary == 'before_pid' and value.get('kind') == 'spawned':
        checkpoint.write_text('ready')
        await asyncio.Event().wait()
    return value
NativePythonWorker._guardian_event = event
original_record = SQLStateStore.record_analysis_evidence
async def record(store, agent_id, evidence):
    await original_record(store, agent_id, evidence)
    if evidence.kind == 'generation':
        metadata.write_text(json.dumps({'facts': evidence.facts.to_dict(), 'run_id': evidence.run_id}))
        if (boundary == 'allocation' and evidence.facts['status'] == 'admitted') or (boundary == 'after_pid' and evidence.facts['status'] == 'running'):
            checkpoint.write_text('ready')
            await asyncio.Event().wait()
SQLStateStore.record_analysis_evidence = record
async def main():
    model = ToolboxAwareMockModelProvider((_tool('crash', 'analysis_execute', {'code':'print(42)', 'expected_state':{'generation':0,'revision':0}}),))
    agent = await Agent.create('recovery', root=root, workspace=workspace_for(root), model=model, model_profile=_profile(model))
    await agent.run('Crash across launch ownership handoff.')
asyncio.run(main())
"""
