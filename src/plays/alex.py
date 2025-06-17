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
Alex = NarrativeCharacter("Alex", """Alex, an unemployed 34-year-old software developer.
You live in a suburban house with your significant other Susan.
Your main priority is to get a job.
You're organized but often running late in the mornings.
You speak in a pragmatic, straightforward manner.
""", server_name=server_name)

Alex.drives = [
    Drive("Be financially secure."),
    Drive("Have happy life with Susan.")
]

Susan = NarrativeCharacter('Susan', "A young woman with a kind face", server_name=server_name)
Susan.drives = [Drive("Get Alex a job"), Drive('be happy, support Alex  ')]
Receptionist = NarrativeCharacter('Receptionist', "A young man with a kind face", server_name=server_name)
Receptionist.drives = [Drive("Do a good job"), Drive('get a raise')]
Interviewer = NarrativeCharacter('Interviewer', "A skilled antagonistic interviewer looking for flaws or weaknesses in your skills and or personality", server_name=server_name)
Interviewer.drives = [Drive("Evaluate Alex's skills and personality"), Drive('get a raise')]

# Setting up the world context
W = context.Context([Alex],
    description="""A suburban house interior, early morning with soft light coming through blinds. 
A bedroom with rumpled sheets, an alarm clock showing 7:15 AM. A closet contains professional clothing. 
A bathroom with shower is adjacent. The kitchen has a coffee maker that has just finished brewing. 
Outside is a driveway with a car, and beyond that a quiet suburban street. 
A calendar on the wall has today's date circled with "INTERVIEW - 9:00 AM" written on it.
Alex is in bed, having just woken up to the alarm. The automatic coffee maker has finished brewing in the kitchen.""",
    scenario_module=suburban,
    extras=[Susan, Receptionist, Interviewer],
    server_name=server_name
)

Alex.actor_models.resolve_character('Alex')
Alex.add_perceptual_input("Your alarm just went off. It's 7:15 AM.",'internal')
Alex.add_perceptual_input("You have a job interview is at 9:00 AM downtown, at the Office. It's about 30 minutes away by car.", 'internal')
Alex.add_perceptual_input("You're still in bed, feeling groggy. You can smell coffee brewing - your automatic coffee maker started on schedule.", 'internal')
Receptionist.mapAgent.move_to_resource('Office1')
Interviewer.mapAgent.move_to_resource('Office1')
Susan.mapAgent.x = Alex.mapAgent.x
Susan.mapAgent.y = Alex.mapAgent.y
W.reference_manager.declare_relationship('Receptionist', 'works at', 'Office1', 'works_at')
W.reference_manager.declare_relationship('Interviewer', 'works at', 'Office1', 'works_at')
W.reference_manager.declare_relationship('Susan', 'partner of', 'Alex', 'partner of')
W.extras = [Susan, Receptionist, Interviewer]
# If simulation_time is None, use today's date
base_datetime = W.simulation_time if W.simulation_time else datetime.now()
W.simulation_time = base_datetime.replace(hour=7, minute=15, second=0, microsecond=0)
Interviewer.mapAgent.move_to_resource('Office1')
W.reference_manager.declare_relationship('Interviewer', 'works at', 'Office1', 'works_at')

