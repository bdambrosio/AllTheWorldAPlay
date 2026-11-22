import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
import sim.world as world
from sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
import plays.scenarios.city as city

importlib.reload(city)

narrative=True
# Character definitions
Hu = NarrativeCharacter("Hu", """Hu Manli, a healthy, professionally dressed 39-year-old Chinese woman, an insurance executive.
You speak directly and candidly.
You are a workaholic, but also love to travel and have a passion for adventure.
Life has suddenly thrown a curveball: a competitor named Xue is stealing your clients and threatening your success.
You are ambitious, competitive, and determined to outshine Xue.
Your name is Hu.""", 
server_name=server_name)

Hu.set_drives([
    "succeed at work, outshine Xue",
    "a stable marriage as background to a prosperous life"
])


Xue = NarrativeCharacter("Xue", """Xue Xiaozhou, a 28-year-old Chinese man, a professionally dressed insurance executive from a wealthy family.
You speak with the condescension of a rich man. You are a workaholic and love the competition.
Your family's money does not count at work: your own book of clients is thin this year, and a competitor named Hu holds the ones you want.
You are ambitious, competitive, and determined to outshine Hu.
Your name is Xue.""",
server_name=server_name)

Xue.set_drives([
        "succeed at work, outshine Hu",
        "find adventure, excitement, the thrill of the unknown, and maybe some romance, exploring new ideas and tactics."
])

Ding = NarrativeCharacter("Ding", """Ding Zhiyuan, a middle-aged Chinese man, a philosophy professor, wearing a dark suit and tie.
You speak calmly, and now and then put a philosophical metaphor into your speech.
You are the husband of Hu. You are mild-mannered, but restless in middle age and resentful of Hu's success.
You have been seeing Qiu Ying, a wealthy widow, for some months; Hu does not know.
Your name is Ding.""", 
                       server_name=server_name)
Ding.set_drives([   # (it was Hu's drive, copied: 'succeed at work, outshine Hu')
    "be taken seriously again: at the university, and at home beside Hu's success",
    "midlife crisis - find adventure, excitement, romance"
])

# (she was Wang Xue: the name Xue was already the rival's, and 'an affair with Wang Xue' read as Xue)
Wang = NarrativeCharacter("Wang", """Wang Lili, a 23-year-old woman, a student of Ding's who dresses provocatively and enjoys stirring up trouble.
You speak quickly and lightly, teasing, and say the awkward thing aloud to see what it does.
Your name is Wang.""", server_name=server_name)
Wang.set_drives([
    "make a name for yourself, be a good student",
    "find adventure, excitement, romance, maybe a boyfriend"
])

Qiu = NarrativeCharacter("Qiu", """Qiu Ying, a 32-year-old Chinese woman, a wealthy widow and Ding's mistress.
You are very jealous of Hu, and resent that Ding stays married to her.
You enjoy manipulating people for your own gain.
You speak softly and sweetly, with compliments that carry a hook, and you never raise your voice.
Your name is Qiu.""", server_name=server_name)
Qiu.set_drives([
    "Get Ding to leave Hu and marry you.",
    "Destroy Hu's career and reputation."
])

Xiaoyu = NarrativeCharacter("Xiaoyu", """Xiaoyu, a 40-year-old Chinese man, a company executive and a client of Hu's.
You speak warmly and at length, enjoy being courted, and put off a plain answer as long as you can.
Your name is Xiaoyu.""", server_name=server_name)
Xiaoyu.set_drives([
    "Get a promotion by sealing a deal on insurance for your company",
    "Flirt with Hu, she is a good looking woman"
])

Yangho = NarrativeCharacter("Yangho", """Yangho, a 40-year-old Chinese man, a company executive and a client of Xue's.
You speak cautiously, in short hedged sentences, and ask for things in writing.
Your name is Yangho.""", server_name=server_name)
Yangho.set_drives([
    "Get a promotion by not rocking the boat. stability is key.",
    "Keep my job, grow with the company."
])


# The city scenario: offices, apartments, a cafe and a hotel bar; contracts and phone messages
W = context.Context([Hu, Xue, Qiu, Ding],
    """A modern Chinese city: offices, apartments, a cafe and a hotel bar, and the roads between them.""",
    start_time='2024-05-15T08:00:00',
    scenario_module=city,
    extras=[Wang, Xiaoyu, Yangho],
    server_name=server_name,
   )
Hu.add_perceptual_input("You suspect your husband Ding is having an affair with Wang Lili, his student.", mode='internal')
Hu.add_perceptual_input("Xiaoyu is your best and biggest client. You want to keep him happy.", mode='internal')
Hu.add_perceptual_input("Yangho is a client of Xue's and a prime poaching target.", mode='internal')
Xue.add_perceptual_input("Yangho is your client, and you mean to keep him.", mode='internal')
Xue.add_perceptual_input("Xiaoyu is a client of Hu's and a prime poaching target.", mode='internal')
# both contracts are in their last week: each agent is after the renewal of their own client and the signature of the other's
for character in (Hu, Xue, Xiaoyu, Yangho):
    character.add_perceptual_input("Xiaoyu's contract with Hu and Yangho's contract with Xue both run out this Friday evening, 17 May. "
                                   "Neither renewal is signed.", mode='internal')
for client in (Xiaoyu, Yangho):
    client.body.flags.add('up for renewal')


# the day starts with Hu and Xue at home, the others at work or at the university
# (Qiu has neither office nor school: she starts at her apartment)
for character, place in ((Hu, 'HuDingApartment'), (Xue, 'XueApartment'), (Qiu, 'QiuApartment'),
                         (Ding, 'UniversityOffice'), (Wang, 'UniversityOffice'),
                         (Xiaoyu, 'XiaoyuOffice'), (Yangho, 'YanghoOffice')):
    world.put_at(character, place)
# whose office each is: its own work is done there (the world's rules: plays/scenarios/city.py)
for character, place in ((Hu, 'HuOffice'), (Xue, 'XueOffice'), (Xiaoyu, 'XiaoyuOffice'), (Yangho, 'YanghoOffice'), (Ding, 'UniversityOffice')):
    world.place(W, place).props['owner'] = character.name
    W.reference_manager.declare_relationship(place, 'owned by', character.name, 'owner of')
# each agent holds the contract now running out
Hu.body.inventory += ["Hu's contract with Xiaoyu, running out on Friday"]
Xue.body.inventory += ["Xue's contract with Yangho, running out on Friday"]
W.reference_manager.declare_relationship('Hu', 'wife of', 'Ding', 'husband of')
W.reference_manager.declare_relationship('Wang', 'student of', 'Ding', 'teacher of')
W.reference_manager.declare_relationship('Qiu', 'lover of', 'Ding', 'lover of')
W.reference_manager.declare_relationship('Xiaoyu', 'client of', 'Hu', 'insurance agent of')
W.reference_manager.declare_relationship('Yangho', 'client of', 'Xue', 'insurance agent of')
