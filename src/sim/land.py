"""A world of places with positions in metres: what sim/world.py asks, answered for a world file
that states a `layout` (plays/scenarios/forest.py). Places hold other places: a region (the
forest, a clearing, a rocky rise) holds the springs, thickets and huts that stand in it. A
position is (x, y) in metres, east and south of where the play's people first stand; which place
it is in is worked out from where the places are.

The rules of nearness are stated once here, in metres (agreed 2026-11-21):
  here       a thing lies at hand in the same small place (radius up to 10), or within 3
  at a place within its radius, or within 3 of a small one
  together   in the same enclosed place, or within 8 in the open: to hand over, to see what
             another holds, to witness, to walk along, for a fire to keep a danger off
  earshot    25, and never through closed walls
  a step     within the same place, or under 10: not told as a walk
  walk over  30 and in sight: gone to in the course of a deed aimed there
  sight      a thing is seen to 25 times its radius under trees, 75 times in the open, three
             times as far from high ground; never through closed walls, nor across a place that
             blocks the view, nor into a place that hides what is in it. A person counts as 2.5.
  shelter    inside a place that is one
  a danger   comes from a source within 60; first stands 30 to 40 off; reaches the one it comes
             for and anyone within 15 of them; is got clear of at 150 and out of its sight"""
import math
import random
import re

HERE, SMALL, TOGETHER, EARSHOT, STEP, WALK_OVER = 3.0, 10.0, 8.0, 25.0, 10.0, 30.0
PERSON, UNDER_TREES, IN_THE_OPEN, FROM_HIGH = 2.5, 25.0, 75.0, 3.0
DANGER_SOURCE, DANGER_REACH, DANGER_FROM, DANGER_CLEAR = 60.0, 15.0, (30.0, 40.0), 150.0
PACE = 70.0                      # metres a minute, where a region says nothing else
MINUTES_AN_EFFORT = 3.0          # what a walk costs a body is counted by these (a tile of the old grid was about this)
BEARINGS = ('East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest', 'North', 'Northeast')


class Place:
    """A place: id (the name prompts use), kind, description, xy, radius, and what is so of it.
    A region is a place that holds others (the forest, a clearing)."""

    def __init__(self, rid, kind, description, xy, radius, **said):
        self.id, self.kind, self.description, self.xy, self.radius = rid, kind, description, (round(float(xy[0]), 1), round(float(xy[1]), 1)), round(float(radius), 1)
        self.region = bool(said.get('region'))
        self.enclosed = bool(said.get('enclosed'))        # walls: no sight or sound through them
        self.open = bool(said.get('open'))                # an enclosed place standing open (a cave's mouth)
        self.shelter = bool(said.get('shelter'))
        self.blocks_view = bool(said.get('blocks_view'))
        self.hides = bool(said.get('hides'))              # what is inside is not seen from outside (a thicket)
        self.overlooks = bool(said.get('overlooks'))
        self.cover = said.get('cover', 'trees')           # a region: 'trees' or 'open'
        self.pace = said.get('pace')                      # a region: metres a minute on foot
        self.props = {}
        self.made = said.get('made')

    @property
    def owner(self):
        return self.props.get('owner')

    def holds(self, xy):
        return _dist(self.xy, xy) <= self.radius

    def __repr__(self):
        return f'<Place {self.id}>'


class Standing:
    """Where a person stands (kept on the person as `mapAgent`, as the grid kept its own)."""

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _a_walk(minutes):
    return f"about a minute's walk" if minutes == 1 else f"about {minutes} minutes' walk"


def bearing(frm, to):
    """Which way `to` lies from `frm`, of eight; 'Current' when it is the same spot."""
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    if math.hypot(dx, dy) < 1.0:
        return 'Current'
    return BEARINGS[int(((math.degrees(math.atan2(dy, dx)) % 360) + 22.5) // 45) % 8]


class Land:
    def __init__(self, context, scenario, seed=None):
        self.context, self.scenario = context, scenario
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.rng = random.Random(self.seed)
        self.table = {}          # id -> Place, in the order the world came to have them (regions too)
        self.counts = {}
        self.kinds = dict(getattr(scenario, 'kinds', {}) or {})
        self.build(scenario.layout)

    # ------------------------------------------------------------------ laying it out
    def _new(self, kind, description, xy, radius=None, name=None, **said):
        about = {**self.kinds.get(kind, {}), **said}
        radius = about.pop('radius', 3.0) if radius is None else radius
        about.pop('radius', None)
        if isinstance(radius, (tuple, list)):
            radius = self.rng.uniform(*radius)
        key = (kind, bool(about.get('region')))      # a park with a fountain may stand in ground called park
        self.counts[key] = self.counts.get(key, 0) + 1
        rid = name or (f'{kind}_ground{self.counts[key]}' if about.get('region') else f'{kind}{self.counts[key]}')
        self.table[rid] = Place(rid, kind, description, xy, radius, **about)
        return self.table[rid]

    def build(self, layout):
        allocations = {str(a['resource_type']).split('.')[-1]: a for a in getattr(self.scenario, 'resource_rules', {}).get('allocations', [])}
        self.scatter = layout.get('scatter', {'within': 700, 'apart': 40})
        for r in layout['regions']:
            for _ in range(r.get('count', 1)):
                xy = (0.0, 0.0) if r.get('whole') else self._around((0.0, 0.0), r.get('from_start', (0, self.scatter['within'])))
                self._new(r['kind'], r.get('description', ''), xy, region=True, **{k: v for k, v in r.items() if k not in ('kind', 'count', 'from_start', 'whole', 'description')})
        self.whole = max((p for p in self.table.values() if p.region), key=lambda p: p.radius)   # the region that holds the rest
        for spec in layout['places']:
            kind = spec['kind']
            told = allocations.get(kind, {})
            for _ in range(spec.get('count', told.get('count', 1))):
                self._new(kind, spec.get('description') or told.get('description', ''), self._spot(spec), name=spec.get('name'))

    def _around(self, centre, span, angle=None):
        r, a = self.rng.uniform(*span), self.rng.uniform(0, 2 * math.pi) if angle is None else angle
        return (centre[0] + r * math.cos(a), centre[1] + r * math.sin(a))

    def _regions(self, kind):
        return [p for p in self.table.values() if p.region and p.kind == kind]

    def _spot(self, spec):
        """A position for a new place, by where its kind is said to stand."""
        apart, within = spec.get('apart', self.scatter['apart']), self.scatter['within']
        hosts = [p for p in self.table.values() if p.kind == (spec.get('inside') or (spec.get('near') or [None])[0])]
        for attempt in range(400):
            if attempt == 200:
                apart /= 3          # a crowded clearing: nearer its neighbours than the rule, sooner than not there
            if spec.get('inside'):      # within a place that holds others (a room's things in a house)
                host = self.rng.choice(hosts)
                xy = self._around(host.xy, (0, host.radius * 0.75))
            elif spec.get('near'):      # beside a place, between so many metres and so many from its edge
                host = self.rng.choice(hosts)
                xy = self._around(host.xy, (host.radius + spec['near'][1], host.radius + spec['near'][2]))
            elif spec.get('foot_of'):     # at the foot of a region, on the side the play's people start from
                rise = self._regions(spec['foot_of'])[0]
                toward = math.atan2(-rise.xy[1], -rise.xy[0]) + self.rng.uniform(-0.5, 0.5)
                xy = self._around(rise.xy, (rise.radius + 20, rise.radius + 60), toward)
            else:
                on = spec.get('on') or {self.whole.kind: 1}
                kind = self.rng.choices(list(on), weights=list(on.values()))[0]
                if kind == self.whole.kind:
                    r = self.rng.uniform(*spec.get('from_start', (0, within))) if spec.get('from_start') else within * self.rng.random()   # denser near where they start
                    a = self.rng.uniform(0, 2 * math.pi)
                    xy = (r * math.cos(a), r * math.sin(a))
                    if self.region_at(xy) is not self.whole:
                        continue
                else:
                    regions = self._regions(kind)
                    if not regions:
                        continue
                    region = self.rng.choice(regions)
                    xy = self._around(region.xy, (0, region.radius * 0.85))
                    lo, hi = spec.get('from_start', (0, float('inf')))
                    if not lo <= _dist(xy, (0.0, 0.0)) <= hi:
                        continue
            if all(_dist(xy, p.xy) >= apart for p in self.table.values() if not p.region and p not in hosts):
                return xy
        return xy

    # ------------------------------------------------------------------ places
    def places(self):
        return [p for p in self.table.values() if not p.region]

    def listed(self):
        return [(p.id, p) for p in self.places()]

    def place(self, rid):
        p = self.table.get(rid) if rid else None
        return p if p is not None and not p.region else None

    def called(self, name):
        low = str(name or '').lower()
        return next((p for p in self.places() if p.id.lower() == low), None)

    def find(self, text):
        rid = str(text or '').strip().replace('#', '')
        return self.place(rid) or self.called(rid)

    def _kinds(self, regions=False):
        return list(dict.fromkeys(p.kind for p in self.table.values() if p.region == regions))

    @staticmethod
    def _word(name):
        return re.sub(r'[\s_]+', '_', str(name or '').strip()).lower()

    def kind_called(self, name):
        return next((k for k in self._kinds() if k.lower() == self._word(name)), None)

    def region_called(self, name):
        return next((k for k in self._kinds(regions=True) if k.lower() == self._word(name)), None)

    def nearest(self, xy, kind=None):
        found = [(_dist(xy, p.xy), p.id, p) for p in self.places() if kind is None or p.kind == kind]
        if not found:
            return None
        d, _, p = min(found, key=lambda f: f[:2])
        return d, p

    def named_in(self, text):
        low = (text or '').lower()
        return [p.id for p in sorted(self.places(), key=lambda p: len(p.id), reverse=True)
                if re.search(rf'(?<![a-z0-9_]){re.escape(p.id.lower())}(?![a-z0-9_])', low)]

    def add_place(self, kind, description, xy):
        return self._new(kind, description or '', xy)

    def has_owner_part(self, place):
        for a in getattr(self.scenario, 'resource_rules', {}).get('allocations', []):
            if str(a['resource_type']).split('.')[-1] == place.kind:
                return a.get('has_npc', False)
        return False

    # ------------------------------------------------------------------ the ground
    def start(self):
        return (0.0, 0.0)

    def region_at(self, xy):
        """The innermost region a position is in (the whole, when in no other)."""
        inside = [p for p in self.table.values() if p.region and p.holds(xy)]
        return min(inside, key=lambda p: p.radius) if inside else self.whole

    def region(self, xy):
        return self.region_at(xy).kind

    def regions(self):
        return self._kinds(regions=True)

    def nearest_ground(self, kind, anchor):
        if self.region(anchor) == kind:
            return anchor
        found = self._regions(kind)
        return min(found, key=lambda p: _dist(anchor, p.xy)).xy if found else None

    def pace(self, xy):
        return self.region_at(xy).pace or PACE

    def inside(self, xy):
        """The place (not a region) a position is in, the smallest; or None."""
        found = [p for p in self.places() if p.holds(xy)]
        return min(found, key=lambda p: p.radius) if found else None

    def enclosure(self, xy):
        """The walled place a position is inside, the smallest; or None."""
        found = [p for p in self.places() if p.enclosed and p.holds(xy)]
        return min(found, key=lambda p: p.radius) if found else None

    def indoors(self, xy):
        return self.enclosure(xy) is not None

    def sheltered(self, xy):
        return any(p.shelter and p.holds(xy) for p in self.places())

    # ------------------------------------------------------------------ nearness
    def span(self, a, b):
        return _dist(a, b)

    def _walled_apart(self, a, b):
        """Is there a closed wall between two positions?"""
        for p in self.places():
            if p.enclosed and not p.open and p.holds(a) != p.holds(b):
                return True
        return False

    def at_place(self, xy, p):
        return _dist(xy, p.xy) <= max(p.radius, HERE)

    def at_its_foot(self, xy, p):
        return self.at_place(xy, p)

    def together(self, a, b):
        if self._walled_apart(a, b):
            return False
        room = self.enclosure(a)
        return _dist(a, b) <= TOGETHER or (room is not None and room.radius <= SMALL and room.holds(b))

    def on_the_spot(self, xy, spot):
        return self.together(xy, spot)

    def lying_here(self, at, xy):
        if any(p.radius <= SMALL and p.holds(at) and p.holds(xy) for p in self.places()):
            return True
        return _dist(at, xy) <= HERE and not self._walled_apart(at, xy)

    def in_earshot(self, a, b):
        return _dist(a, b) <= EARSHOT and not self._walled_apart(a, b)

    def a_step(self, a, b):
        return _dist(a, b) < STEP or any(p.holds(a) and p.holds(b) for p in self.places())

    def can_walk_over(self, a, b, cfg=None):
        return _dist(a, b) <= WALK_OVER and self.sees(a, b)

    def sees(self, a, b, radius=PERSON, into=None):
        """Can what is at `b`, of this radius, be seen from `a`? into: the place being looked at,
        whose own walls and cover do not hide it."""
        d = _dist(a, b)
        for p in self.places():
            if p is into:
                continue
            if p.enclosed and not p.open and p.holds(a) != p.holds(b):
                return False
            if p.hides and p.holds(b) and not p.holds(a):
                return False
            if p.blocks_view and not p.holds(a) and not p.holds(b) and self._crosses(a, b, p):
                return False
        for p in self.table.values():
            if p.region and p.blocks_view and not p.holds(a) and not p.holds(b) and self._crosses(a, b, p):
                return False
        here, there = self.region_at(a), self.region_at(b)
        reach = radius * (IN_THE_OPEN if here.cover == 'open' and there.cover == 'open' else UNDER_TREES)
        return d <= reach * (FROM_HIGH if here.overlooks else 1.0)

    @staticmethod
    def _crosses(a, b, p):
        """Does the line from a to b pass through the round of a place?"""
        ax, ay, bx, by, (cx, cy) = a[0], a[1], b[0], b[1], p.xy
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / length))
        return math.hypot(ax + t * dx - cx, ay + t * dy - cy) < p.radius

    def in_sight(self, xy, p):
        return self.at_place(xy, p) or self.sees(xy, p.xy, radius=p.radius, into=p)

    def in_range(self, xy, p):
        return True

    def in_the_neighbourhood(self, xy, p, within=None):
        return self.in_sight(xy, p)

    # ------------------------------------------------------------------ walking, and distance in words
    def walk(self, a, b):
        minutes = _dist(a, b) / self.pace(b)
        return minutes, minutes / MINUTES_AN_EFFORT

    def minutes(self, a, b):
        return max(1, round(_dist(a, b) / self.pace(b)))

    def told_minutes(self, n):
        return max(1, round(n))

    def far(self, a, b):
        d = _dist(a, b)
        return 'a few steps' if d < 10 else f'about {round(d, -1):.0f} metres' if d < 100 else _a_walk(self.minutes(a, b))

    def away(self, a, b):
        return f'{self.far(a, b)} {bearing(a, b).lower()}'

    def sight_words(self, a, b):
        return self.away(a, b)

    def reach_words(self, a, b, name):
        d = _dist(a, b)
        return (f'a few steps from {name}' if d < 10 else f'close by {name}' if d < 50
                else f"{_a_walk(self.minutes(a, b))} from {name}")

    def where(self, xy):
        room = self.inside(xy)
        if room is not None:
            return f'at {room.id}'
        found = self.nearest(xy)
        if found is None:
            return f'in the {self.region(xy).lower()}'
        d, p = found
        if self.at_place(xy, p):
            return f'at {p.id}'
        way = bearing(p.xy, xy).lower()
        return (f'a few steps {way} of {p.id}' if d < 10 else f'close by {p.id}, to its {way}' if d < 50
                else f"{_a_walk(self.minutes(xy, p.xy))} {way} of {p.id}")

    def resolve(self, location, anchor):
        name = str(location or '').strip().strip('#"\'.')
        for suffix in (' terrain', ' resource', ' area'):
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)].strip()
        if name.lower() in ('', 'same', 'same place', 'same location', 'same as before', 'here', 'unchanged'):
            return None
        p = self.called(name)
        if p is not None:
            return p.xy, p.id
        kind = self.kind_called(name)
        if kind is not None:
            found = self.nearest(anchor, kind)
            if found:
                return found[1].xy, found[1].id
        ground = self.region_called(name)
        if ground is not None:
            xy = self.nearest_ground(ground, anchor)
            if xy is not None:
                return xy, ground
        return None

    # ------------------------------------------------------------------ dangers
    def danger_start(self, xy, rng):
        r, a = rng.uniform(*DANGER_FROM), rng.uniform(0, 2 * math.pi)
        return (xy[0] + r * math.cos(a), xy[1] + r * math.sin(a))

    def danger_reaches(self, xy, other):
        return _dist(xy, other) <= DANGER_REACH

    def danger_source_near(self, xy, p, within=None):
        return _dist(xy, p.xy) <= DANGER_SOURCE

    def danger_clear(self, xy, danger_at, clear=None):
        return self.sheltered(xy) or (_dist(xy, danger_at) >= DANGER_CLEAR and not self.sees(danger_at, xy))

    def danger_words(self, xy, danger_at):
        d = _dist(xy, danger_at)
        return 'close by' if d <= 10 else 'a short way off' if d <= 50 else 'some way off'

    def apart_words(self, positions):
        spread = max(_dist(a, b) for a in positions for b in positions)
        return 'side by side' if spread <= TOGETHER else 'a few steps apart' if spread <= 25 else 'some way apart' if spread <= 100 else 'far apart'

    def danger_picture_words(self, positions, danger_at):
        near = min(_dist(xy, danger_at) for xy in positions)
        return 'close by' if near <= 10 else 'nearby' if near <= 50 else 'in the distance'

    # ------------------------------------------------------------------ what is in view
    def in_view(self, who, xy):
        """(places, people) in view from a position: each place with how far and which way."""
        people = []
        for c in self.context.everyone():
            agent = getattr(c, 'mapAgent', None)
            if c is who or agent is None:
                continue
            other = (agent.x, agent.y)
            if self.sees(xy, other):
                people.append((_dist(xy, other), c.name, other))
        seen = [(_dist(xy, p.xy), p.id, p) for p in self.places() if self.in_sight(xy, p)]
        return sorted(seen, key=lambda s: s[:2]), sorted(people)

    def surroundings(self, who):
        xy = (who.mapAgent.x, who.mapAgent.y)
        seen, people = self.in_view(who, xy)
        ground = self.region_at(xy)
        head = f"You are {self.where(xy)}, " + ('indoors' if self.indoors(xy) else f'in the {ground.kind.lower()}' + (', under cover' if self.sheltered(xy) else '')) + '.'
        def told(d, to):
            return 'here' if d < HERE else f"{self.far(xy, to)} {bearing(xy, to).lower()}"
        lines = [head]
        if people:
            lines.append('People in view: ' + '; '.join(f'{name} ({told(d, at)})' for d, name, at in people) + '.')
        if seen:
            lines.append('Places in view: ' + '; '.join(f"{rid} ({'here' if self.at_place(xy, p) else told(d, p.xy)})" for d, rid, p in seen[:24]) + '.')
        summary = head + (f" You see {', '.join(name for _, name, _ in people)}." if people else '') \
            + (f" In view: {', '.join(rid for _, rid, _ in seen[:12])}." if seen else '')
        return '\n'.join(lines), [rid for _, rid, _ in seen], [name for _, name, _ in people], [], summary

    def view_report(self, who):
        return self.surroundings(who)[0]
