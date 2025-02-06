from sim.memory.core import Drive
import sim.worldsim as worldsim
import sim.agh as agh
from sim.context import Context
server='deepseeklocal'
#server='local'
J = agh.Agh("Jean", """You are Jean Macquart, a hardworking young peasant farmer. 
You left military service to return to the family farm.
You are strong, honest and committed to working the land, but have a quick temper.
You speak plainly and directly, in the style of a 19th century french peasant speaking to an acquaintance.
You hope to inherit a share of the family farm and make a living as a farmer.
Despite being french, you speak in 19th century peasant English.
Your name is Jean.""", server=server)
J.drives = [Drive("maintaining and working the family farm"),
Drive("gaining your rightful inheritance - justice and fairness in how the land is divided"),
Drive("finding love and a wife to build a family with"),
Drive("immediate needs of survival - food, shelter, health, rest from backbreaking labor")
]
#J.add_to_history("You think – Another long day of toil in the fields. When will I get my fair share of this land that I pour my sweat into? I returned from the army to be a farmer, not a lackey for my family.")
J.add_to_history("You think - That Francoise is a hard worker, and pretty too. If I ever had my own farm she would be a good partner.")
F = agh.Agh("Francoise", """You are Francoise Fouan, an attractive young woman from a neighboring peasant family.
You are hardworking and stoic, accustomed to the unending labor required on a farm.
You conceal your feelings and speak carefully, knowing every word will be gossiped about in the village.
You dream of marrying and having a farm of your own to manage one day.
You speak plainly and directly, in the style of a 19th century french peasant speaking to an acquaintance.
Despite being french, you speak in 19th century peasant English.
Your name is Francoise.""", server=server)
F.drives = [Drive("finding a good husband to marry, gaining status and security"),
Drive("avoiding scandal and protecting your reputation"),
Drive("helping your family with the endless chores"),
Drive("brief moments of rest and simple joys amid the hardships")
]
F.add_to_history("You think – I saw that Jean Macquart again in the field. He works so hard for his family. Seems to have a chip on his shoulder though. Best not to stare and set the gossips' tongues wagging.")


W =Context([J, F],
"""A small 19th century French farming village surrounded by fields ripe with wheat and other crops. 
It is late afternoon on a hot summer day.""", server=server)
#worldsim.IMAGEGENERATOR = 'tti_serve'
#worldsim.main(W)
