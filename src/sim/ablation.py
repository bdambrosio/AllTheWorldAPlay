"""Feature switches for ablation runs (tools/batch_run.py).

The set of disabled features comes with the load_play command ({"ablate": "stylizer,stance"})
and lasts until the next load. Everything is enabled by default.

  commitments  post-dialog commitment extraction
  stance       LLM emotional stance (characters keep the default stance)
  drive_appraisal  drives that move with events (sim/cognitive/drives.py); off, they rest at their baselines
"""
FEATURES = ('commitments', 'stance', 'drive_appraisal')
_disabled: set = set()


def set_disabled(spec):
    """spec: comma-separated feature names, a list, or None."""
    global _disabled
    names = spec.split(',') if isinstance(spec, str) else (spec or [])
    names = {n.strip() for n in names if n and n.strip()}
    unknown = names - set(FEATURES)
    if unknown:
        raise ValueError(f'unknown ablation feature(s) {sorted(unknown)}; known: {FEATURES}')
    _disabled = names


def enabled(feature):
    return feature not in _disabled


def disabled():
    return sorted(_disabled)
