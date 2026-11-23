"""Structured run trace for scoring a performance (tools/score_play.py).

One JSONL file per loaded play, under ~/.local/share/alltheworldaplay/logs/traces/.
Every record has: kind, t (wall clock), run, sim_time, act, scene, plus the event's fields.

    from sim.trace import trace
    trace.event('act', character=self.name, mode=..., action=...)

Logging only: a closed trace makes event() a no-op, and a failure inside event() is
swallowed, so tracing can never change or break a performance.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path.home() / '.local' / 'share' / 'alltheworldaplay' / 'logs' / 'traces'


class PlayTrace:
    def __init__(self):
        self._file = None
        self._lock = threading.Lock()
        self.context = None
        self.run_id = None
        self.path = None

    @property
    def active(self):
        return self._file is not None

    def open(self, play_name: str, context) -> Path:
        self.close()
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(play_name).stem
        self.run_id = f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.path = TRACE_DIR / f'{self.run_id}.jsonl'
        self.context = context
        self._pictures = 0
        self._file = open(self.path, 'a', buffering=1)
        self.event('run_start', play=play_name)
        return self.path

    def close(self):
        if self._file is None:
            return
        self.event('run_end')
        with self._lock:
            try:
                self._file.close()
            finally:
                self._file = None
                self.context = None

    def _position(self):
        ctx = self.context
        if ctx is None:
            return None, None, None
        sim_time = getattr(ctx, 'simulation_time', None)
        act = getattr(ctx, 'current_act', None)
        scene = getattr(ctx, 'current_scene', None)
        return (sim_time.isoformat() if isinstance(sim_time, datetime) else sim_time,
                act.get('act_number') if isinstance(act, dict) else None,
                scene.get('scene_number') if isinstance(scene, dict) else None)

    def prompt(self, prompt, response, **fields):
        """Every model call of the run, as sent and as answered, in <run>.prompts.jsonl beside the
        trace: tag, owner, where the play stood, the messages, the reply. The only record of
        prompts was simulation.log, which each engine start overwrites; a line nobody could
        account for ('photographs of Hu and Ding', who are husband and wife) could not be traced
        to what the model had been told. Read with tools/prompts.py."""
        if self._file is None or self.path is None:
            return
        try:
            sim_time, act, scene = self._position()
            record = {'t': round(time.time(), 3), 'sim_time': sim_time, 'act': act, 'scene': scene,
                      'tag': fields.get('tag'), 'owner': fields.get('owner'), 'secs': fields.get('secs'),
                      'finish_reason': fields.get('finish_reason'), 'prompt': prompt, 'response': response}
            with self._lock:
                with open(self.path.with_suffix('.prompts.jsonl'), 'a') as f:
                    f.write(json.dumps(record, default=str) + '\n')
        except Exception:
            pass  # logging must never affect the play

    def keep(self, path, of):
        """A picture the run has just made, kept with the run: the engine writes its pictures to a few files
        it writes over in turn, so that of a finished play only the last were left, and the book of a
        performance (tools/book.py) had to make the rest again. A picture of the stage or of a deed (of:
        'world', 'act') is kept each time; a portrait (of: the character's name) is kept as the latest of
        that character. Named in the trace as an 'image_file' record."""
        if self._file is None or not path:
            return
        try:
            folder = TRACE_DIR.parent / 'pictures' / self.run_id
            folder.mkdir(parents=True, exist_ok=True)
            if of in ('world', 'act'):
                self._pictures += 1
                kept = folder / f'{self._pictures:03d}_{of}.png'
            else:
                kept = folder / ('portrait_' + re.sub(r'\W+', '_', str(of)) + '.png')
            shutil.copyfile(path, kept)
            self.event('image_file', of=of, file=str(kept))
        except Exception:
            pass  # keeping a picture must never affect the play

    def event(self, kind: str, **fields):
        if self._file is None:
            return
        try:
            sim_time, act, scene = self._position()
            record = {'kind': kind, 't': round(time.time(), 3), 'run': self.run_id,
                      'sim_time': sim_time, 'act': act, 'scene': scene}
            record.update(fields)
            line = json.dumps(record, default=str)
            with self._lock:
                if self._file is not None:
                    self._file.write(line + '\n')
        except Exception:
            pass  # tracing must never affect the play


trace = PlayTrace()


class TraceErrorHandler(logging.Handler):
    """Copies ERROR records of the engine logger into the trace."""
    def emit(self, record):
        trace.event('error', logger=record.name, message=record.getMessage()[:1000])


def llm_call_listener(**fields):
    """Registered with utils.llm_api.add_call_listener (utils does not import sim). The call's
    figures go to the trace; what was sent and what came back go to the run's prompt log."""
    prompt, response = fields.pop('prompt', None), fields.pop('response', None)
    trace.event('llm_call', **fields)
    trace.prompt(prompt, response, **fields)
