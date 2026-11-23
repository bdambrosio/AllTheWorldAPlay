"""A thread file (src/sim/thread.py) for a performance played before threads were kept, rebuilt from its trace.

    python tools/thread_from_trace.py <run id> [--no-digest] [--out DIR]

The engine writes a thread at the end of a part from its own state. A run that ended before that
has only its trace, which is a log of changes: this reads it forward and keeps the last word on
each thing. It is a lesser record, and says so ('source': 'trace', and 'not_rebuilt' lists what the
trace does not hold). What each person carries forward is written by the same prompt as in the
engine, from the lines of the scenes that person was in and their own notes on what they tried.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from sim import thread

TRACES = os.path.expanduser('~/.local/share/alltheworldaplay/logs/traces')
NOT_REBUILT = ["each person's own memories (the digests are written from the lines of the scenes they were in)",
               'where a thing lies or who holds it after it was first placed, except what each was last pictured holding',
               "the needs of a person's body other than those the trace reported changing",
               'what has been told of each life as the ledger kept it (the play\'s planted lines stand in for it)',
               'traces on places that a later deed ended, and whether each promise was kept']


def rebuild(run):
    rows = [json.loads(l) for l in open(os.path.join(TRACES, f'{run}.jsonl'))]
    of = lambda kind: [r for r in rows if r['kind'] == kind]
    last = lambda kind: (of(kind) or [{}])[-1]
    people, order = {}, []
    def person(name, lead=False, description=''):
        if name not in people:
            order.append(name)
            people[name] = {'name': name, 'lead': lead, 'description': ' '.join(description.split()), 'lately': '', 'wants': [], 'at': None,
                            'place': '', 'needs': {}, 'holds': [], 'sheltered': False, 'flags': [], 'to_the_eye': [],
                            'does_not_know_the_name_of': [], 'carries_forward': {}}
        return people[name]
    for c in last('play_loaded').get('characters') or []:
        person(c['name'], True, c.get('description', ''))['wants'] = [{'text': d} for d in c.get('drives') or []]
    behind = last('director').get('behind', '')
    for r in of('part_cast'):
        described = next((l.split('|')[2].strip() for l in behind.splitlines() if l.startswith('#person') and l.split('|')[0].split()[-1] == r['name']), '')
        p = person(r['name'], False, f"{r['name']}, {described}" if described else r['name'])
        p['wants'], p['place'] = [{'text': w} for w in r.get('wants') or []], r.get('place') or ''
        if r.get('unknown'):
            leads = [n for n in order if people[n]['lead']]
            p['does_not_know_the_name_of'] = list(leads)
            for n in leads:
                people[n]['does_not_know_the_name_of'].append(r['name'])
    for r in of('strangers'):
        for n in r['cast']:
            people[n]['does_not_know_the_name_of'] += [m for m in r['cast'] if m != n]
    places = {p['id']: {'id': p['id'], 'kind': p.get('of_kind'), 'description': p.get('description', ''), 'at': p.get('at'), 'owner': None,
                        'made': None, 'names': [], 'props': {}} for p in last('world_built').get('places') or [] if not p.get('region')}
    things, states, traces, promised, standing, models, cast_of, acts = {}, [], [], [], [], {}, {}, {}
    for r in rows:
        k, who = r['kind'], r.get('character')
        if k == 'name_learned' and who in people:
            people[who]['does_not_know_the_name_of'] = [n for n in people[who]['does_not_know_the_name_of'] if n not in r['names']]
        elif k == 'drives' and who in people:
            people[who]['wants'] = [{'text': t, 'activation': round(a, 2)} for t, a in r['activations'].items()]
        elif k == 'drive_evolved' and who in people:
            for w in people[who]['wants']:
                if w['text'] == r['was']:
                    w['text'] = r['now']
        elif k in ('need_changed', 'body') and who in people:
            people[who]['needs'][r['need']] = round(r.get('now', r.get('level', 0)), 2)
        elif k in ('travel', 'walk') and who in people:
            if r.get('to'):
                people[who]['at'] = r['to']
            if r.get('place'):
                people[who]['place'] = r['place']
                for other in r.get('party') or []:
                    people[other]['place'] = r['place']
        elif k in ('scene_place', 'scene_moved'):
            for n in r.get('movers') or cast_of.get(r.get('scene'), []):
                if n in people:
                    people[n]['place'], people[n]['at'] = r['place'], r.get('at') or (places.get(r['place']) or {}).get('at')
        elif k == 'scene_start':
            cast_of[r['scene']] = [n for n in r.get('cast') or [] if n in people]
        elif k == 'state_left' and who in people:
            states.append({'who': who, 'words': r['state'], 'placed': bool(r.get('placed')), 'since': r['sim_time'], 'scene': r.get('scene')})
        elif k == 'state_ended':
            states[:] = [s for s in states if not (s['who'] == who and s['words'].lower() == r['state'].lower())]
        elif k == 'mark_left':
            traces.append({'words': r['mark'], 'on': r.get('on'), 'place': people.get(who, {}).get('place', ''), 'by': who})
        elif k == 'mark_ended':
            traces[:] = [t for t in traces if t['words'].lower() != r['mark'].lower()]
        elif k == 'promise' and r.get('when') in ('standing', 'later'):
            promised.append({'who': r['who'], 'what': r['what'], 'kind': r['when'], 'since': r['sim_time']})
        elif k == 'stands':
            standing.append({'fact': r['fact'], 'about': r.get('about') or [], 'known': r.get('known') or [], 'since': r['sim_time'], 'ended': None})
        elif k == 'stands_ended':
            for s in standing:
                if s['fact'] == r['fact']:
                    s['ended'] = r['sim_time']
        elif k == 'thing_added':
            things[r['name']] = {'name': r['name'], **({'holder': r['holder']} if r.get('holder') else {})}
        elif k in ('thing_handled', 'thing_changed', 'thing_modelled') and r.get('thing'):
            things.setdefault(r['thing'], {'name': r['thing']}).update(state=r.get('state'))
        elif k == 'thing_gone':
            things.pop(r.get('thing'), None)
        elif k == 'model_kept':
            models[r['name']] = r['model']
        elif k == 'place_added':
            places[r['id']] = {'id': r['id'], 'kind': r.get('of_kind') or r.get('type'), 'description': r.get('description', ''), 'at': r.get('xy'),
                               'owner': None, 'made': True, 'names': [r['name']] if r.get('name') else [], 'props': {}}
        elif k == 'part_cast' and r.get('owns') and r.get('place') in places:
            places[r['place']]['owner'] = r['name']
        elif k == 'act_start':
            acts[r['number']] = {'number': r['number'], 'kind': '', 'title': r.get('title', ''), 'summary': '', 'choices': []}
        elif k == 'decision' and r.get('act') in acts:
            acts[r['act']]['choices'].append({'scene': r.get('scene'), 'character': who, 'decision': r['decision'], 'choice': r['choice']})
        elif k == 'scene_end' and r.get('act') in acts:
            acts[r['act']]['summary'] += ('; ' if acts[r['act']]['summary'] else '') + '; '.join(o['outcome'] for o in r.get('outcomes') or [] if o.get('test') is True)
        elif k == 'image' and isinstance(r.get('facts'), dict) and (r['facts'].get('person') or {}).get('name') in people:
            people[r['facts']['person']['name']]['holds'] = list(r['facts']['person'].get('holding') or [])
        elif k == 'clues_placed':
            for c in r.get('clues') or []:
                things.setdefault(c['thing'], {'name': c['thing']}).update(place=c['place'], hidden=bool(c.get('hidden')), **({'fixed': True} if c.get('fixed') else {}))
    final = max((n for n in cast_of if isinstance(n, int)), default=None)   # how a person was placed is over when they move off: only the last scene's holds
    states[:] = [{k: v for k, v in s.items() if k != 'scene'} for s in states if not s['placed'] or s['scene'] == final]
    for s in states:
        people[s['who']]['to_the_eye'].append(s['words'])
    episode, question, known = last('episode'), last('director_question').get('question') or last('central_question').get('question', ''), last('director_read').get('known') or {}
    clues = [{**{k: v for k, v in c.items() if k != 'hidden'}, 'now': {'holder': things.get(c['thing'], {}).get('holder'), 'place': c['place'], 'hidden': bool(c.get('hidden'))}}
             for c in last('clues_placed').get('clues') or []]
    end = max((r['sim_time'] for r in rows if r.get('sim_time')), default=None)
    part = {'part': 1, 'run': run, 'written': None, 'source': 'trace', 'not_rebuilt': NOT_REBUILT, 'clock': end[:19] if end else None,
            'setting': ' '.join(str(last('play_loaded').get('setting', '')).split()),
            'questions': [{'text': question, 'raised': 1, 'settled': bool(episode.get('facts')), 'come_to': known, 'closed': None}] if question else [],
            'so': {'facts': episode.get('facts') or [], 'clues': clues, 'marks': episode.get('marks') or []},
            'story': {'acts': [acts[n] for n in sorted(acts)], 'planted': last('threads').get('threads') or []},
            'transcript': [f"{(r.get('sim_time') or '')[11:16]} {r['speaker']}: {r['text']}" if r.get('speaker') else f"{(r.get('sim_time') or '')[11:16]} {r['text'].strip()}"
                           for r in rows if r['kind'] == 'transcript_line' and r.get('scene') is not None and r.get('line_kind') in ('speech', 'narration')
                           and r.get('text', '').strip() and not r['text'].lstrip().startswith('---')],
            'people': [people[n] for n in order],
            'stage': {'things': list(things.values()), 'traces': traces, 'states': states, 'told': [], 'promised': promised, 'standing': standing,
                      'writings': {}, 'readers': {}, 'shared': [], 'danger': None},
            'world': {'scenario': None, 'seed': last('world_built').get('seed'), 'places': list(places.values()), 'models': models}}
    return rows, cast_of, part


def witnessed(rows, cast_of, name):
    """The lines of the scenes this person was in, and their own notes on what they tried."""
    seen, notes = [], []
    for r in rows:
        if r['kind'] == 'transcript_line' and name in cast_of.get(r.get('scene'), []) and r.get('text', '').strip() and not r['text'].lstrip().startswith('-----scene'):
            at = (r.get('sim_time') or '')[11:16]
            seen.append(f"{at} {r['speaker']}: {r['text']}" if r.get('speaker') else f"{at} {r['text'].strip()}")
        elif r['kind'] == 'tried' and r.get('character') == name:
            notes.append(f"{(r.get('sim_time') or '')[11:16]} (with {r.get('other')}) {r['note']}")
    return '\n'.join(seen), '\n'.join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--no-digest', action='store_true', help='leave what each person carries forward empty (no model is called)')
    ap.add_argument('--out', help='write the thread file here, not among the threads the engine keeps')
    a = ap.parse_args()
    rows, cast_of, part = rebuild(a.run)
    prompts = {}
    if not a.no_digest:
        from utils import llm_api
        from utils.Messages import UserMessage
        import plays.config as config
        llm = llm_api.LLM(config.plan_server_name or config.server_name, getattr(config, 'plan_model_name', None))
        for p in part['people']:
            seen, notes = witnessed(rows, cast_of, p['name'])
            if not seen:
                continue
            prompts[p['name']] = thread.digest_prompt(p['name'], p['description'], seen, notes)
            reply = llm.ask({}, [UserMessage(content=prompts[p['name']])], tag='thread.digest', temp=0.3, stops=['</end>'], max_tokens=900) or ''
            p['carries_forward'] = thread.read_digest(reply)
            print(f"{p['name']}: {len(seen.splitlines())} lines witnessed, digest {'made' if p['carries_forward'].get('lived') else 'FAILED'}")
    record = {'thread': a.run, 'play': next((r.get('play') for r in rows if r['kind'] == 'run_start'), ''), 'parts': [part]}
    if a.out:
        thread.DIR = a.out
    where = thread.save(record)
    if prompts:
        with open(where.replace('.json', '.digest-prompts.json'), 'w') as f:
            json.dump(prompts, f, indent=1, ensure_ascii=False)
    print(where)


if __name__ == '__main__':
    main()
