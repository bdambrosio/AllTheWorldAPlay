"""The ledger: everything on the stage that changes, in one place.

The map says where places are and what kind each is, and no performance changes it. What does
change is kept here: each thing as an object of its own, with who holds it or where it lies,
how it has been left, and what the world has come to know of its kind (sim/keeper.py); and how
the people on stage stand to one another (posture, position, contact, a thing held out, a thing
shared), what written things say and who has read them.

It replaces a list of names in each body's hands, a list of things lying about, and seven
dictionaries on the context, each with its own lifetime and none of them visible to every
reader. (Two branches shared one state because a state was kept by name; a fire was a flag
that nothing could show.)

The older ways in still work, as views of this:
  body.inventory     the names of what a body holds; append and remove move things in the ledger.
                     A name it gives back knows which thing it is (Name.thing).
  context.props      the things lying about, each a Thing (a dict: 'name', 'at', and where they
                     apply 'hidden', 'fixed', 'spot', 'state', 'since', 'kind')
  context.held_out, .shared, .poses, .between, .writings, .readers

One way in. Whatever changes the stage is one event of the play (a deed, a line, a walk, a scene
being set, time passing, the director placing a thing), and it is committed: Context.committing
opens the event, the change is made inside it, and the event keeps what it touched beside the line
shown for it. A thing knows the event that brought it into the play (thing.made; None: it was
there before the play) and the one that last changed it (thing.last), so whoever describes it can
say whose doing it is. A change made outside any event is a path that has not been brought in yet:
it is traced once per place in the code ('uncommitted_write') and counted (Ledger.loose_writes).
"""
import sys

from sim.trace import trace


def _site():
    """Where in the code a change came from: the nearest caller outside this module."""
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    if frame is None:
        return '?'
    return f"{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_code.co_name}:{frame.f_lineno}"


class Thing(dict):
    """One thing. 'name'; then 'holder' (a name) or 'at' ((x, y)); and as they apply: 'hidden',
    'fixed', 'spot', 'state', 'since', 'kind'. Two things of one name are two things."""
    __hash__ = object.__hash__
    ledger = None   # the ledger it is kept in
    made = None     # the event that brought it into the play (None: there before the play)
    last = None     # the event that last moved or changed it
    state_by = None # the event that left it in the state it is in

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def _note(self):
        if self.ledger is not None:
            self.ledger.wrote('thing', self)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._note()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._note()

    def pop(self, *args):
        had = args[0] in self
        value = super().pop(*args)
        if had:
            self._note()
        return value

    def update(self, *args, **more):
        super().update(*args, **more)
        self._note()

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        super().__setitem__(key, default)
        self._note()
        return default


def _changing(base, method):
    def changed(self, *args, **more):
        out = getattr(base, method)(self, *args, **more)
        self.ledger.wrote(self.what)
        return out
    changed.__name__ = method
    return changed


class Kept(list):
    """A list the ledger keeps (things held out, shared, how people stand to one another): a
    change to it is a change to the stage."""
    def __init__(self, ledger, what, items=()):
        super().__init__(items)
        self.ledger, self.what = ledger, what

    def __iadd__(self, items):
        self.extend(items)
        return self


for _m in ('append', 'extend', 'insert', 'remove', 'pop', 'clear', '__setitem__', '__delitem__', 'sort', 'reverse'):
    setattr(Kept, _m, _changing(list, _m))


class KeptMap(dict):
    """A dictionary the ledger keeps (postures, what written things say, who has read them)."""
    def __init__(self, ledger, what):
        super().__init__()
        self.ledger, self.what = ledger, what

    def pop(self, *args):
        had = args[0] in self
        value = dict.pop(self, *args)
        if had:
            self.ledger.wrote(self.what)
        return value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


for _m in ('__setitem__', '__delitem__', 'update', 'clear', 'popitem'):
    setattr(KeptMap, _m, _changing(dict, _m))


class Name(str):
    """The name of a held thing, as body.inventory gives it, with the thing it names."""
    thing = None

    def __new__(cls, thing):
        name = super().__new__(cls, thing['name'])
        name.thing = thing
        return name


class Ledger:
    def __init__(self):
        self.things = []      # [Thing]
        self.held_out = Kept(self, 'held_out')    # (item, holder, to): held out to someone in this scene, theirs to take
        self.shared = Kept(self, 'shared')        # (item, {names}): one thing two people both have
        self.poses = KeptMap(self, 'poses')       # name -> posture
        self.between = Kept(self, 'between')      # (a, words, b, 'position' | 'contact')
        self.writings = KeptMap(self, 'writings') # name of a written thing -> what it says
        self.readers = KeptMap(self, 'readers')   # name of a written thing -> who has read it
        # what a deed has left on a place: {'words', 'at', 'on' (a thing's name or None), 'event'}. A
        # trace is not a thing: it is never listed among what lies there, only as someone's doing
        self.marks = Kept(self, 'marks')
        # what each person has told of their own life before the play: {'who', 'to' [names], 'fact',
        # 'event'}. A thing said of oneself is kept as told, and is the same for every later reader
        self.told = Kept(self, 'told')
        # what has been promised or agreed in a line, to keep or for later: {'who', 'to' [names],
        # 'what', 'kind' ('standing' | 'later'), 'event'}. (Kept only in the trace, an agreement
        # made in one scene was struck from the next one's set-up as never made.)
        self.promised = Kept(self, 'promised')
        self.open = []          # the events being committed, the innermost last (Context.committing)
        self.begun = False      # the play has begun: from now on every change belongs to an event
        self.loose_writes = {}  # where in the code a change was made outside any event -> how often

    # -- one way in
    def wrote(self, what, thing=None):
        """Something of the stage has changed (`thing`, or `what`: 'poses', 'needs', 'position' ...).
        It belongs to the event being committed; outside one, once the play has begun, it is a
        path not yet brought in."""
        if self.open:
            event = self.open[-1]
            if thing is not None:
                if not any(t is thing for t in event['things']):
                    event['things'].append(thing)
                thing.last = event
            elif what not in event['changed']:
                event['changed'].append(what)
        elif self.begun:
            site = _site()
            self.loose_writes[site] = self.loose_writes.get(site, 0) + 1
            if self.loose_writes[site] == 1:
                trace.event('uncommitted_write', what=what, site=site, **({'thing': thing.get('name')} if thing is not None else {}))

    # -- things
    def held(self, holder):
        return [t for t in self.things if t.get('holder') == holder]

    def loose(self):
        return [t for t in self.things if t.get('at') is not None]

    def add(self, name, holder=None, at=None, **more):
        thing = Thing(name=name, **{k: v for k, v in more.items() if v not in (None, False)})
        thing.ledger = self
        thing.made = self.open[-1] if self.open else None
        self.place(thing, holder=holder, at=at)
        self.things.append(thing)
        self.wrote('thing', thing)
        return thing

    def place(self, thing, holder=None, at=None):
        """Put a thing in someone's hands or down at a spot; it is in one place only."""
        thing.pop('holder', None)
        thing.pop('at', None)
        if holder is not None:
            thing['holder'] = holder
            thing.pop('hidden', None)
            thing.pop('spot', None)
        elif at is not None:
            thing['at'] = tuple(at)
        return thing

    def put(self, thing, holder=None, at=None):
        """A thing (one taken out of a hand or picked up, perhaps) goes into a hand or down at a spot, the same thing still."""
        if not any(t is thing for t in self.things):
            self.things.append(thing)
            thing.ledger = self
        self.wrote('thing', thing)
        return self.place(thing, holder=holder, at=at)

    def remove(self, thing):
        if any(t is thing for t in self.things):
            self.things[:] = [t for t in self.things if t is not thing]
            self.wrote('thing', thing)
            return True
        return False


def of(context):
    """The context's ledger, made when first asked for."""
    found = context.__dict__.get('_ledger')
    if found is None:
        found = context.__dict__['_ledger'] = Ledger()
    return found


# ---------------------------------------------------------------------- the older ways in
class Inventory:
    """What a body holds, as the list of names it always was. Reading gives Names; adding and
    removing move things in the ledger. Before a body is bound to a ledger it is a plain list."""
    def __init__(self, ledger, holder):
        self._ledger, self._holder = ledger, holder

    def _things(self):
        return self._ledger.held(self._holder)

    def __iter__(self):
        return iter([Name(t) for t in self._things()])

    def __len__(self):
        return len(self._things())

    def __bool__(self):
        return bool(self._things())

    def __contains__(self, name):
        return any(t['name'] == name for t in self._things())

    def __getitem__(self, i):
        return [Name(t) for t in self._things()][i]

    def __eq__(self, other):
        return list(self) == list(other)

    def __repr__(self):
        return repr([t['name'] for t in self._things()])

    def __add__(self, other):
        return list(self) + list(other)

    def __iadd__(self, names):
        for name in names:
            self.append(name)
        return self

    def count(self, name):
        return sum(1 for t in self._things() if t['name'] == name)

    def index(self, name):
        return [t['name'] for t in self._things()].index(name)

    def append(self, name):
        thing = name if isinstance(name, Thing) else getattr(name, 'thing', None)
        if thing is not None:   # a thing changing hands is the same thing, with what is known of it
            return self._ledger.put(thing, holder=self._holder)
        return self._ledger.add(str(name), holder=self._holder)

    def remove(self, name):
        thing = getattr(name, 'thing', None)
        if thing is None or thing.get('holder') != self._holder:
            thing = next((t for t in self._things() if t['name'] == name), None)
        if thing is None:
            raise ValueError(f'{name!r} is not held by {self._holder}')
        self._ledger.remove(thing)
        return Name(thing)   # out of this hand; append it elsewhere and it is the same thing

    def thing(self, name):
        """The first held thing of this name, or None."""
        return next((t for t in self._things() if t['name'] == name), None)


class Loose(list):
    """The things lying about, as the list it always was: append and remove reach the ledger."""
    def __init__(self, ledger):
        super().__init__(ledger.loose())
        self._ledger = ledger

    def append(self, thing):
        if not isinstance(thing, Thing) or not any(t is thing for t in self._ledger.things):
            thing = self._ledger.add(thing['name'], at=thing['at'], **{k: v for k, v in thing.items() if k not in ('name', 'at', 'holder')})
        super().append(thing)

    def remove(self, thing):
        self._ledger.remove(thing)
        super().remove(thing)


def props(context):
    return Loose(of(context))


def set_props(context, things):
    """context.props = [...]: the things lying about are now these."""
    ledger = of(context)
    keep = [t for t in things if isinstance(t, Thing)]
    for t in ledger.loose():
        if not any(t is k for k in keep):
            ledger.remove(t)
    for t in things:
        if not isinstance(t, Thing):
            ledger.add(t['name'], at=t['at'], **{k: v for k, v in t.items() if k not in ('name', 'at', 'holder')})
        elif not any(t is x for x in ledger.things):
            ledger.things.append(t)
            t.ledger = ledger
            ledger.wrote('thing', t)


def install(cls):
    """Give a context class the older attributes as views of its ledger."""
    cls.props = property(props, set_props)
    for attr in ('held_out', 'shared', 'poses', 'between', 'writings', 'readers'):
        def get(self, attr=attr):
            return getattr(of(self), attr)

        def put(self, value, attr=attr):
            kept = getattr(of(self), attr)   # the same kept list or dictionary, with new contents
            if value is kept:
                return
            if isinstance(kept, dict):
                value = dict(value)
                kept.clear()
                kept.update(value)
            else:
                kept[:] = list(value)
        setattr(cls, attr, property(get, put))
    return cls
