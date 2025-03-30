# Custom terrain types for suburban environment
import asyncio
from enum import Enum
import importlib
import sys, os
import wave
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sim.context as context
import sim.agh as agh
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
from sim.scenarios import suburban

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
importlib.reload(suburban)

class SuburbanTerrain(Enum):
    House = 1
    Yard = 2
    Street = 3
    Sidewalk = 4
    Park = 5
    OfficeBuilding = 6

# Resources that might be found in this environment
class SuburbanResource(Enum):
    Refrigerator = 1
    Shower = 2
    Closet = 3
    Car = 4
    BusStop = 5
    Mailbox = 6
    Coffee = 7
    Breakfast = 8

# Main character - person with morning routine and job
Alex = agh.Character("Alex", """You are Alex, an unemployed 34-year-old software developer.
You live in a suburban house with your significant other Susan.
You're organized but often running late in the mornings.
You speak in a pragmatic, straightforward manner.
""", server_name=server_name)

Alex.drives = [
    Drive("Be financially secure."),
    Drive("get to the job interview on time and making a good impression, and get the job"),
    Drive("maintaining basic needs: hygiene, food, and appropriate appearance"),
    #Drive("managing anxiety about the interview"),
    Drive("keeping your home in reasonable order")
]


# Setting up the world context
W = context.Context([Alex],
    description="""A suburban house interior, early morning with soft light coming through blinds. 
A bedroom with rumpled sheets, an alarm clock showing 7:15 AM. A closet contains professional clothing. 
A bathroom with shower is adjacent. The kitchen has a coffee maker that has just finished brewing. 
Outside is a driveway with a car, and beyond that a quiet suburban street. 
A calendar on the wall has today's date circled with "INTERVIEW - 9:00 AM" written on it.
Alex is in bed, having just woken up to the alarm. The automatic coffee maker has finished brewing in the kitchen.""",
    scenario_module=suburban,
    server_name=server_name
)

Alex.actor_models.resolve_character('Alex')
Alex.add_perceptual_input("Your alarm just went off. It's 7:15 AM.",'internal')
Alex.add_perceptual_input("You have a job interview is at 9:00 AM downtown, at the Office. It's about 30 minutes away by car.", 'internal')
Alex.add_perceptual_input("You're still in bed, feeling groggy. You can smell coffee brewing - your automatic coffee maker started on schedule.", 'internal')
receptionist =W.get_npc_by_name('Receptionist', description="A young man with a kind face", x=20, y=20, create_if_missing=True)
receptionist.mapAgent.move_to_resource('Office#1')
W.reference_manager.declare_relationship('Receptionist', 'works at', 'Office#1', 'works_at')

interviewer = W.get_npc_by_name('Interviewer', description="A middle-aged man with an alert, questioning face", x=20, y=20, create_if_missing=True)
interviewer.mapAgent.move_to_resource('Office#1')
W.reference_manager.declare_relationship('Interviewer', 'works at', 'Office#1', 'works_at')

