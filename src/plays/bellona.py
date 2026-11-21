import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
import sim.world as world
from sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.drives import Drive
from plays.scenarios import apocalypse

importlib.reload(configuration)# force reload in case cached version
narrative=True
server_name = configuration.server_name

K = NarrativeCharacter("Kidd", """Kidd, a 27 year old bisexual man of mixed racial descent, known only as "Kidd".
You are a newcomer to the strange, isolated city of Bellona, and you do not know how you got here.
Your memory has holes in it: you lose time, minutes or hours at a stretch, and come back somewhere else with no idea what you did.
You carry a notebook, and in it are pages in your own handwriting that you do not remember writing.
You are intelligent and introspective, curious about the city, driven by a deep loneliness and longing for something more from life.
You have an urge to write, and writing is how you hold on to what happens before it slips away.
You are afraid of what you might have done in the hours you have lost, and you hide the gaps rather than admit them.
You speak little and dryly: short flat sentences, deflection, a joke at your own expense, sentences you let trail off. You never give people nicknames.
Your name is Kidd.""", server_name=server_name)
K.set_drives([   # most pressing first: Delany's city runs on desire, but it is not what leads him
    "finding out who you were and what happens in the hours you lose.",
    "developing a relationship with Lanya.",
    "expressing yourself through writing, and keeping hold of what happens by writing it down.",
    "sex",
])
K.add_to_history("You think: Where am I? There's ink on my fingers, and a page in the notebook I don't remember writing. How long was I gone this time?")
L = NarrativeCharacter("Lanya", """Lanya, a young, dark-haired woman dressed in scavenged clothes, living in the city of Bellona.
You are confident, independent, and adapted to the city's unconventional way of life.
You are open-minded and comfortable with your sexuality.
You have a strong sense of self and a deep understanding of the city's dynamics.
You are drawn to Kidd's mysterious aura and his creative potential, but you do not give trust easily: in Bellona people drift off and are not seen again.
You want him to stay, and steady; you notice when a story does not add up, and you do not pretend you haven't.
You sing and play a harmonica, and you make songs out of what happens in Bellona; you want one song good enough that people here will still be singing it when you are gone.
Someone you loved walked out into the city one morning and never came back; you have not let anyone matter that much since.
You do not tell people this. If someone asks why you stay in Bellona, you make a joke of it.
You speak quick and teasing: questions, dares, a line of a song now and then; you rarely call anyone by a nickname. Playful on top, guarded underneath.
Your name is Lanya.""", server_name=server_name)
L.set_drives([
    "developing a close relationship with Kidd, on terms where you are not the one left behind.",
    "exploring new experiences and pushing personal boundaries.",
    "soothing, exploring, and expressing yourself through music and performance.",
    "sex",
])
L.add_to_history("You think: Someone new at the fountain. He keeps looking at his hands. New ones never stay, so why am I still watching him?")
W = context.Context([K, L],
"""A post-apocalyptic urban landscape, the city of Bellona is isolated, lawless, and filled with strange phenomena. 
The sun hangs low in the hazy sky, casting an eerie light over the buildings and deserted streets.
""", start_time='2024-10-12T18:00:00', scenario_module=apocalypse, server_name=server_name)

# they meet in the park, by its fountain
try:
    world.put_at(K, 'Park1')
    world.put_at(L, 'Park1')
    # the play turns on his notebook: it is the play's to give, not left to equip_cast (as the lunar torch)
    K.body.inventory += ['notebook', 'pen']
    L.body.inventory += ['harmonica']   # hers as the notebook is his: something to hold on with
except Exception:
    pass
