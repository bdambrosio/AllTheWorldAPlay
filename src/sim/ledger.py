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
"""


class Thing(dict):
    """One thing. 'name'; then 'holder' (a name) or 'at' ((x, y)); and as they apply: 'hidden',
    'fixed', 'spot', 'state', 'since', 'kind'. Two things of one name are two things."""
    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other


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
        self.held_out = []    # (item, holder, to): held out to someone in this scene, theirs to take
        self.shared = []      # (item, {names}): one thing two people both have
        self.poses = {}       # name -> posture
        self.between = []     # (a, words, b, 'position' | 'contact')
        self.writings = {}    # name of a written thing -> what it says
        self.readers = {}     # name of a written thing -> who has read it

    # -- things
    def held(self, holder):
        return [t for t in self.things if t.get('holder') == holder]

    def loose(self):
        return [t for t in self.things if t.get('at') is not None]

    def add(self, name, holder=None, at=None, **more):
        thing = Thing(name=name, **{k: v for k, v in more.items() if v not in (None, False)})
        self.place(thing, holder=holder, at=at)
        self.things.append(thing)
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
        return self.place(thing, holder=holder, at=at)

    def remove(self, thing):
        if any(t is thing for t in self.things):
            self.things[:] = [t for t in self.things if t is not thing]
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


def install(cls):
    """Give a context class the older attributes as views of its ledger."""
    cls.props = property(props, set_props)
    for attr in ('held_out', 'shared', 'poses', 'between', 'writings', 'readers'):
        def get(self, attr=attr):
            return getattr(of(self), attr)

        def put(self, value, attr=attr):
            setattr(of(self), attr, value)
        setattr(cls, attr, property(get, put))
    return cls
