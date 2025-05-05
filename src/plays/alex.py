from datetime import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
from plays.scenarios import suburban

importlib.reload(configuration)  # force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(suburban)

map_file_name = 'suburban.py' # needed to trigger narrative creation

# Main character - person with morning routine and job
Alex = NarrativeCharacter("Alex", """You are Alex, an unemployed 34-year-old software developer.
You live in a suburban house with your significant other Susan.
Your main priority is to get a job.
You're organized but often running late in the mornings.
You speak in a pragmatic, straightforward manner.
""", server_name=server_name)

Alex.drives = [
    Drive("Be financially secure."),
    Drive("Have happy life with Susan.")
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

# If simulation_time is None, use today's date
base_datetime = W.simulation_time if W.simulation_time else datetime.now()
W.simulation_time = base_datetime.replace(hour=7, minute=15, second=0, microsecond=0)

interviewer = W.get_npc_by_name('Interviewer', description="A middle-aged man with an alert, questioning face", x=20, y=20, create_if_missing=True)
interviewer.mapAgent.move_to_resource('Office#1')
W.reference_manager.declare_relationship('Interviewer', 'works at', 'Office#1', 'works_at')

