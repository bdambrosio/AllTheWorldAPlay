"""A performance as a small book: the permanent record of one night's play.

    .venv/bin/python tools/book.py [RUN] [--render] [--out DIR]     (default RUN: the latest)

Reads the run's trace (sim/trace.py) and nothing else of the engine, and writes, under
~/.local/share/alltheworldaplay/books/<run>/:
    book.html    the whole book in one file, pictures embedded
    book.pdf     the same, paged at 5.5 x 8.25 inches (printed by headless Chrome, if installed)
    pictures/    the pictures, and pictures.json saying where each came from

The text is the performance as it was played, set as a book and not rewritten: an act is a part,
a scene is a chapter on a new page, the set-up opens it, deeds are told and lines are spoken
under their speaker's name. Nothing is asked of a model. What the engine says to its operator
(the markers between scenes, the running summaries) is left out; what it told the audience (a
danger coming on, hours passing, a walk) is kept.

Pictures. A run that kept its pictures (an 'image_file' record for each) uses them. An older run
did not: its scene pictures went to eight files written over in turn, so only the last few are
left. Those are taken (matched by the time they were written) and copied into the book's folder,
where they stay. With --render the lost ones are made again by the picture server from the
prompt the trace recorded for each: close to what was shown, not the same picture. The colophon
says which are which.
"""
import base64, html, io, json, os, re, shutil, subprocess, sys, time
from datetime import datetime

HOME = os.path.expanduser('~/.local/share/alltheworldaplay')
TRACES = os.path.join(HOME, 'logs', 'traces')
BOOKS = os.path.join(HOME, 'books')
IMAGES = os.path.expanduser('~/.local/share/AllTheWorld/images')   # where the engine writes a picture (utils/llm_api.py)
IMAGE_URL = os.environ.get('IMAGE_URL', 'http://127.0.0.1:5008/generate_image')
NUMBERS = ('One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen',
           'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen', 'Twenty')


def number(n):
    return NUMBERS[n - 1] if 0 < n <= len(NUMBERS) else str(n)


def latest():
    runs = sorted((f for f in os.listdir(TRACES) if f.endswith('.jsonl') and not f.endswith('.prompts.jsonl')),
                  key=lambda f: os.path.getmtime(os.path.join(TRACES, f)))
    return runs[-1][:-len('.jsonl')] if runs else None


def read(run):
    rows = []
    with open(os.path.join(TRACES, run + '.jsonl')) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


# ---------------------------------------------------------------- the performance, in order

def performance(rows):
    """{'play', 'began', 'question', 'players' [{'name', 'says'}], 'acts' [{'number', 'title', 'scenes', 'coda'}]};
    a scene is {'number', 'title', 'where', 'setup', 'lines' [{'kind', 'who', 'text', 't'}]}."""
    start = next((r for r in rows if r['kind'] == 'run_start'), {})
    book = {'play': os.path.splitext(start.get('play') or 'a play')[0], 'began': start.get('t'), 'question': '', 'players': [], 'acts': []}
    acts, scenes = {}, {}
    for r in rows:
        if r['kind'] == 'act_start':
            acts[r['number']] = {'number': r['number'], 'title': r.get('title') or '', 'scenes': [], 'coda': ''}
            book['acts'].append(acts[r['number']])
        elif r['kind'] == 'scene_start' and r.get('act') in acts:
            scenes[r['scene']] = {'number': r['scene'], 'title': r.get('title') or '', 'where': '', 'setup': (r.get('pre_narrative') or '').strip(),
                                  'lines': [], 't': r['t'], 'sim_time': r.get('sim_time')}
            acts[r['act']]['scenes'].append(scenes[r['scene']])
        elif r['kind'] == 'central_question' and not book['question']:
            book['question'] = r.get('question') or ''
    last_act = None
    for r in rows:
        if r['kind'] != 'transcript_line':
            continue
        kind, who, text = r.get('line_kind'), r.get('speaker') or '', (r.get('text') or '').strip()
        last_act = acts.get(r.get('act'), last_act)
        if kind == 'other':   # before the curtain: each lead in its own words, and the play's question
            if who == 'Prologue':
                book['question'] = book['question'] or text
            elif who:
                book['players'].append({'name': who, 'says': text})
            continue
        if kind == 'narration':
            if text.startswith('coda:'):   # what the act came to, told when it closed
                if last_act is not None:
                    last_act['coda'] = text[len('coda:'):].strip()
                continue
            marker = re.match(r'-+\s*scene\s*-+.*?\n\s*Setting:\s*(.*)', text, re.S)
            if marker and r.get('scene') in scenes:
                scenes[r['scene']]['where'] = marker.group(1).strip()
                continue
            if text.startswith('----') or r.get('scene') not in scenes:   # the engine to its operator
                continue
            if ' '.join(text.split()) == ' '.join(scenes[r['scene']]['setup'].split()):
                continue
        if r.get('scene') in scenes and text:
            scenes[r['scene']]['lines'].append({'kind': kind, 'who': who, 'text': text, 't': r['t']})
    for name in sorted({l['who'] for s in scenes.values() for l in s['lines'] if l['who']} - {p['name'] for p in book['players']},
                       key=lambda n: min(l['t'] for s in scenes.values() for l in s['lines'] if l['who'] == n)):
        first = min((s for s in scenes.values() if any(l['who'] == name for l in s['lines'])), key=lambda s: s['t'])
        book['players'].append({'name': name, 'says': '', 'enters': first['number']})   # no word of its own before the curtain: where it comes in
    return book


# ---------------------------------------------------------------- the pictures

def _jpeg(src, dest):
    from PIL import Image
    im = Image.open(src) if not isinstance(src, bytes) else Image.open(io.BytesIO(src))
    im.convert('RGB').save(dest, 'JPEG', quality=88)


def _rendered(prompt, size='768x768'):
    import requests
    try:
        r = requests.get(IMAGE_URL, params={'prompt': prompt, 'size': size}, timeout=120)
        return r.content if r.status_code == 200 else None
    except requests.RequestException:
        return None


def pictures(run, rows, folder, render=False, say=print):
    """Bring the run's pictures into the book's folder, once: {'scenes': [{'of', 'act', 'scene', 't', 'file', 'source'}],
    'portraits': {name: file}}. Kept in pictures.json, so a later making of the book finds them where the first left them."""
    os.makedirs(folder, exist_ok=True)
    note = os.path.join(folder, 'pictures.json')
    had = json.load(open(note)) if os.path.exists(note) else {'scenes': [], 'portraits': {}}
    known = {(p['t'], p['of']): p for p in had['scenes'] if os.path.exists(os.path.join(folder, p['file']))}
    kept = [r for r in rows if r['kind'] == 'image_file']                 # a run that kept its own pictures
    made = kept or [r for r in rows if r['kind'] == 'image']
    ends = max((r['t'] for r in rows), default=0)
    rotating = [(os.path.getmtime(os.path.join(IMAGES, f)), os.path.join(IMAGES, f)) for f in (os.listdir(IMAGES) if os.path.isdir(IMAGES) else ())
                if re.fullmatch(r'worldsim(_\d+)?\.png', f)]
    out = {'scenes': [], 'portraits': dict(had['portraits'])}
    for n, r in enumerate(p for p in made if p.get('of') in ('world', 'act')):
        p = known.get((r['t'], r['of']))
        if p is None:
            name, source = f'{n:02d}.jpg', None
            if r.get('file') and os.path.exists(r['file']):
                _jpeg(r['file'], os.path.join(folder, name)); source = 'performance'
            else:   # written a few seconds after it was asked for, and not yet written over
                after = [(m, f) for m, f in rotating if 0 <= m - r['t'] < 45]
                later = [x for x in made if x.get('of') in ('world', 'act') and x['t'] > r['t']]
                if after and not any(x['t'] < min(after)[0] for x in later):
                    _jpeg(min(after)[1], os.path.join(folder, name)); source = 'performance'
                elif render and r.get('prompt'):
                    say(f'  rendering again: picture {n + 1} (act {r.get("act")}, scene {r.get("scene")})')
                    data = _rendered(r['prompt'])
                    if data:
                        _jpeg(data, os.path.join(folder, name)); source = 'rendered again'
            if source is None:
                continue
            p = {'of': r['of'], 'act': r.get('act'), 'scene': r.get('scene'), 't': r['t'], 'file': name, 'source': source}
        out['scenes'].append(p)
    for r in made:   # a character's latest portrait
        name = r.get('of')
        if name in ('world', 'act') or not name:
            continue
        src = r.get('file') if r.get('file') and os.path.exists(r['file']) else os.path.join(IMAGES, f'{name}.png')
        fresh = kept or (os.path.exists(src) and r['t'] <= os.path.getmtime(src) <= ends + 120)
        if os.path.exists(src) and fresh:
            file = 'portrait_' + re.sub(r'\W+', '_', name) + '.jpg'
            _jpeg(src, os.path.join(folder, file)); out['portraits'][name] = file
    json.dump(out, open(note, 'w'), indent=1)
    return out


# ---------------------------------------------------------------- the page

def words(text):
    """Text for the page: escaped, apostrophes and quotation marks curled, *sung or stressed* in italics."""
    t = html.escape(' '.join(str(text).split()), quote=False)
    t = re.sub(r"(?<=\w)'(?=\w)", '’', t)
    t = re.sub(r'"([^"]*)"', r'“\1”', t)
    t = re.sub(r"(^|[\s(\[—])'", r'\1‘', t).replace("'", '’')
    return re.sub(r'\*([^*]+)\*', r'<em>\1</em>', t)


def spoken(text):
    t = text.strip()
    if len(t) > 1 and t[0] == t[-1] and t[0] in '\'"':
        t = t[1:-1]
    return '“' + words(t) + '”'


def told(line):
    """A deed as a sentence: the trace keeps it without its doer's name where it began with it."""
    t = line['text'].strip()
    t = t if not line['who'] or t[:1].isupper() or t.startswith(line['who']) else f"{line['who']} {t}"
    return words(t if t[-1:] in '.!?…”’' else t + '.')


def when(scene):
    """'the park, 2024-10-12 18:00 (dusk)' as 'The park · Saturday, dusk'."""
    m = re.match(r'(.*),\s*(\d{4}-\d\d-\d\d) (\d\d:\d\d)\s*\((.*)\)\s*$', scene['where'])
    if not m:
        return words(scene['where'])
    place, day, hour, light = m.groups()
    return f"{words(place[:1].upper() + place[1:])} &middot; {datetime.strptime(day, '%Y-%m-%d').strftime('%A')}, {words(light)}, {hour}"


def figure(folder, picture, cls):
    data = base64.b64encode(open(os.path.join(folder, picture['file'] if isinstance(picture, dict) else picture), 'rb').read()).decode()
    return f'<figure class="{cls}"><img alt="" src="data:image/jpeg;base64,{data}"></figure>'


def chapter(scene, shown, folder):
    out = [f'<section class="chapter" id="scene{scene["number"]}">']
    head = next((p for p in shown if p['of'] == 'world' and p.get('scene') == scene['number']), None)
    if head:
        out.append(figure(folder, head, 'plate'))
    out.append(f'<p class="count">Chapter {number(scene["number"])}</p><h2>{words(scene["title"])}</h2>')
    if scene['where']:
        out.append(f'<p class="where">{when(scene)}</p>')
    inline = sorted((p for p in shown if p['of'] == 'act' and p.get('scene') == scene['number']), key=lambda p: p['t'])
    lines, i, opened = scene['lines'], 0, False
    before = [l for l in lines if l['kind'] == 'narration' and l['t'] < scene['t']]   # told before the curtain rose on it (a danger coming on)
    for l in before:
        out.append(f'<p class="aside">{words(l["text"])}</p>')
    lines = [l for l in lines if l not in before]
    if scene['setup']:
        out.append(f'<p class="setup">{words(scene["setup"])}</p><p class="rule">&#10086;</p>')
    while i < len(lines):
        l = lines[i]
        while inline and inline[0]['t'] <= l['t']:
            out.append(figure(folder, inline.pop(0), 'inset'))
        first = '' if opened else ' first'
        if l['kind'] == 'narration':
            out.append(f'<p class="aside">{words(l["text"])}</p>')
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if l['kind'] == 'action' and nxt and nxt['kind'] == 'speech' and nxt['who'] == l['who']:   # done and said in one breath
            out.append(f'<p class="line{first}">{told(l)} {spoken(nxt["text"])}</p>')
            i += 2
        elif l['kind'] == 'action':
            out.append(f'<p class="line{first}">{told(l)}</p>')
            i += 1
        elif l['kind'] == 'thought':
            out.append(f'<p class="line thought{first}"><span class="who">{words(l["who"])}</span>, to herself or himself: <em>{words(l["text"].strip(". "))}</em></p>'
                       .replace(', to herself or himself:', ' thinks:'))
            i += 1
        else:
            out.append(f'<p class="line said{first}"><span class="who">{words(l["who"])}</span> {spoken(l["text"])}</p>')
            i += 1
        opened = True
    out += [figure(folder, p, 'inset') for p in inline]
    out.append('</section>')
    return '\n'.join(out)


CSS = """
@page { size: 5.5in 8.25in; margin: 0.8in 0.7in 0.85in; @bottom-center { content: counter(page); font: 9pt 'Noto Serif', Georgia, serif; color: #6b6259; } }
@page plain { @bottom-center { content: none; } }
:root { --ink: #26211c; --soft: #6b6259; --paper: #fbf7ef; --accent: #8a3b2e; }
* { box-sizing: border-box; }
html { background: #e9e2d4; }
body { margin: 0; color: var(--ink); font: 17px/1.6 'Noto Serif', 'Iowan Old Style', Georgia, serif; }
section { background: var(--paper); max-width: 46rem; margin: 1.5rem auto; padding: 3rem clamp(1.2rem, 6vw, 4.5rem); box-shadow: 0 1px 8px rgba(0,0,0,.18); }
.plate, .title h1, .title p { max-width: 34rem; margin-left: auto; margin-right: auto; }
.epigraph, .part { min-height: 60vh; }
h1, h2, .count, .who, .part h2 { font-family: 'Noto Serif Display', 'Noto Serif', Georgia, serif; }
h1 { font-size: 34pt; line-height: 1.1; font-weight: 600; margin: .4in 0 .12in; text-align: center; letter-spacing: .01em; }
h2 { font-size: 19pt; line-height: 1.2; font-weight: 600; margin: 0 0 .1in; text-align: center; color: var(--ink); }
.count { text-align: center; font-size: 9.5pt; letter-spacing: .28em; text-transform: uppercase; color: var(--accent); margin: .1in 0 .08in; }
.where { text-align: center; font-size: 9.5pt; color: var(--soft); letter-spacing: .04em; margin: 0 0 .22in; }
p { margin: 0; orphans: 2; widows: 2; }
.setup { font-style: italic; color: #3d3630; }
.setup::first-letter { font: 600 3.1em/0.8 'Noto Serif Display', Georgia, serif; float: left; padding: .06em .08em 0 0; color: var(--accent); font-style: normal; }
.rule { text-align: center; color: var(--accent); margin: .14in 0 .12in; font-size: 13pt; }
.line { text-indent: 1.3em; hyphens: auto; }
.line.first { text-indent: 0; }
.said { margin-top: .03in; }
.who { font-variant: small-caps; letter-spacing: .06em; font-weight: 600; color: var(--accent); margin-right: .15em; }
.aside { font-style: italic; text-align: center; color: var(--soft); margin: .12in .3in; font-size: .93em; }
figure { margin: 0 0 .2in; break-inside: avoid; text-align: center; }
figure img { width: 100%; border-radius: 3px; display: block; }
.plate img { aspect-ratio: 16 / 10; object-fit: cover; }
.inset { margin: .18in auto; width: 72%; }
.inset img { aspect-ratio: 4 / 3; object-fit: cover; }
.title { text-align: center; }
.title .plate img { aspect-ratio: 1 / 1; }
.title .sub { font-size: 10.5pt; color: var(--soft); letter-spacing: .06em; }
.title .house { margin-top: .5in; font-size: 9pt; letter-spacing: .25em; text-transform: uppercase; color: var(--accent); }
.epigraph { display: flex; flex-direction: column; justify-content: center; }
.epigraph p { font-style: italic; font-size: 1.2em; line-height: 1.6; text-align: center; margin: 0 .2in; }
.players h2, .colophon h2, .close h2 { margin-bottom: .25in; }
.player { display: flex; gap: .18in; align-items: flex-start; margin-bottom: .22in; break-inside: avoid; }
.player img { width: 1.15in; height: 1.15in; object-fit: cover; border-radius: 50%; flex: none; }
.player .who { display: block; font-size: 13pt; margin-bottom: .03in; }
.player p { font-size: .92em; line-height: 1.45; }
.part { display: flex; flex-direction: column; justify-content: center; text-align: center; }
.part .count { font-size: 12pt; }
.part h2 { font-size: 25pt; }
.close p, .colophon p { margin-bottom: .1in; }
.close p { font-style: italic; }
.colophon p { font-size: .9em; line-height: 1.5; color: var(--soft); }
@media print {
  html { background: none; }
  body { font-size: 10pt; line-height: 1.5; }
  .plate, .title h1, .title p { max-width: none; }
  section { box-shadow: none; margin: 0; padding: 0; max-width: none; background: none; break-before: page; }
  section.title, section.epigraph, section.part { page: plain; }
  section.title { break-before: auto; }
  .aside, .player { break-inside: avoid; }
  .epigraph, .part { min-height: 6.2in; }
}
"""


def page(book, shown, folder, run):
    title = book['play'][:1].upper() + book['play'][1:]
    began = datetime.fromtimestamp(book['began']) if book['began'] else None
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>']
    front = next((p for p in shown['scenes'] if p['of'] == 'world' and p.get('scene') is None), None)
    out.append('<section class="title">' + (figure(folder, front, 'plate') if front else '') + f'<h1>{words(title)}</h1>'
               + f'<p class="sub">A play in {number(len(book["acts"])).lower()} acts, as performed' + (f' on {began.strftime("%-d %B %Y")}' if began else '') + '</p>'
               + '<p class="house">All the World a Play</p></section>')
    if book['question']:
        out.append(f'<section class="epigraph"><p>{words(book["question"])}</p></section>')
    if book['players']:
        out.append('<section class="players"><p class="count">The Players</p><h2>Who is in it</h2>')
        for p in book['players']:
            file = shown['portraits'].get(p['name'])
            img = f'<img alt="" src="data:image/jpeg;base64,{base64.b64encode(open(os.path.join(folder, file), "rb").read()).decode()}">' if file else ''
            out.append(f'<div class="player">{img}<div><span class="who">{words(p["name"])}</span><p>{words(p["says"]) or f"Comes into the play in Chapter {number(p['enters'])}."}</p></div></div>')
        out.append('</section>')
    for act in book['acts']:
        out.append(f'<section class="part"><p class="count">Act {number(act["number"])}</p><h2>{words(act["title"])}</h2></section>')
        out += [chapter(s, shown['scenes'], folder) for s in act['scenes']]
        if act['coda']:
            out.append(f'<section class="close"><p class="count">Act {number(act["number"])}</p><h2>What it came to</h2><p>{words(act["coda"])}</p></section>')
    scenes = [s for a in book['acts'] for s in a['scenes']]
    said = sum(1 for s in scenes for l in s['lines'] if l['kind'] == 'speech')
    again = [p for p in shown['scenes'] if p['source'] == 'rendered again']
    out.append('<section class="colophon"><p class="count">Colophon</p><h2>About this book</h2>'
               f'<p>This is the record of one performance of <em>{words(title)}</em>: run {html.escape(run)}'
               + (f', which began at {began.strftime("%H:%M on %-d %B %Y")}' if began else '') + '. The play was improvised by its actors from a sketch '
               'and a world; nothing in it was written beforehand, and it will not be played the same way again.</p>'
               f'<p>It ran to {len(book["acts"])} acts and {len(scenes)} scenes, with {said} spoken lines. The text is set here as it was played: '
               'the deeds as the world told them and the lines as they were spoken, in order, with nothing rewritten.</p>'
               f'<p>Of its {len(shown["scenes"])} pictures, {len(shown["scenes"]) - len(again)} are those shown during the performance'
               + (f' and {len(again)} were made again afterwards from the same descriptions, the first having been lost' if again else '') + '.</p>'
               f'<p>Made {datetime.now().strftime("%-d %B %Y")} by tools/book.py.</p></section>')
    out.append('</body></html>')
    return '\n'.join(out)


def printed(html_path, pdf_path):
    """book.pdf from book.html by headless Chrome; the path, or None where no Chrome is installed."""
    chrome = next((c for c in ('google-chrome', 'chromium', 'chromium-browser', 'chrome') if shutil.which(c)), None)
    if chrome is None:
        return None
    for flags in ([], ['--no-sandbox']):
        try:
            subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', *flags, f'--print-to-pdf={pdf_path}', 'file://' + html_path],
                           capture_output=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError):
            continue
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            return pdf_path
    return None


def make(run=None, render=False, out=None, say=print):
    """The book of a run: {'run', 'folder', 'html', 'pdf' (None without Chrome), 'acts', 'scenes', 'pictures'}."""
    run = run or latest()
    if not run or not os.path.exists(os.path.join(TRACES, run + '.jsonl')):
        raise FileNotFoundError(f'no trace of a run called {run!r} under {TRACES}')
    rows = read(run)
    book = performance(rows)
    if not any(a['scenes'] for a in book['acts']):
        raise ValueError(f'{run} has no scene played yet: there is nothing to make a book of')
    folder = os.path.abspath(out or os.path.join(BOOKS, run))
    shown = pictures(run, rows, os.path.join(folder, 'pictures'), render=render, say=say)
    html_path, pdf_path = os.path.join(folder, 'book.html'), os.path.join(folder, 'book.pdf')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(page(book, shown, os.path.join(folder, 'pictures'), run))
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    return {'run': run, 'folder': folder, 'html': html_path, 'pdf': printed(html_path, pdf_path), 'acts': len(book['acts']),
            'scenes': sum(len(a['scenes']) for a in book['acts']), 'pictures': len(shown['scenes']),
            'rendered_again': sum(1 for p in shown['scenes'] if p['source'] == 'rendered again')}


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    where = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    if where in args:
        args.remove(where)
    t = time.time()
    made = make(args[0] if args else None, render='--render' in sys.argv, out=where)
    print(f"{made['run']}: {made['acts']} acts, {made['scenes']} scenes, {made['pictures']} pictures ({made['rendered_again']} rendered again), {time.time() - t:.0f}s")
    print(made['html'])
    print(made['pdf'] or '(no PDF: no Chrome found to print it)')
