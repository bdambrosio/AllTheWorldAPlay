from datetime import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
from sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.drives import Drive
from plays.scenarios import suburban

importlib.reload(configuration)  # force reload in case cached version

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(suburban)

map_file_name = 'suburban.py' # needed to trigger narrative creation
narrative=True
# Main character - person with morning routine and job
Alex = NarrativeCharacter("Alex", """Alex, an unemployed 34-year-old man, a software developer.
You live in a suburban house with your significant other Susan.
Your main priority is to get a job.
You're organized but often running late in the mornings.
You speak in a pragmatic, straightforward manner.
Your name is Alex.""", server_name=server_name)

Alex.drives = [
    Drive("Be financially secure."),
    Drive("Have happy life with Susan.")
]

Susan = NarrativeCharacter('Susan', "Susan, a young woman with a kind face, Alex's partner. You believe in him more than he does himself, and you are a little tired of saying so. Your name is Susan.", server_name=server_name)
Susan.drives = [Drive("Get Alex a job"), Drive('be happy, support Alex  ')]
Receptionist = NarrativeCharacter('Receptionist', "The receptionist at the Office, a young man, polite and a little bored, who has seen a lot of nervous candidates this week.", server_name=server_name)
Receptionist.drives = [Drive("Do a good job"), Drive('get a raise')]
Interviewer = NarrativeCharacter('Interviewer', "The interviewer at the Office, a man in his forties, skilled and antagonistic, who looks for the flaws in a candidate's skills and personality.", server_name=server_name)
Interviewer.drives = [Drive("Evaluate Alex's skills and personality"), Drive('get a raise')]

# Setting up the world context
W = context.Context([Alex],
    description="""A suburban house interior, early morning with soft light coming through blinds. 
A bedroom with rumpled sheets, an alarm clock showing 6:45 AM. A closet contains professional clothing. 
A bathroom with shower is adjacent. The kitchen has a coffee maker that has just finished brewing. 
Outside is a driveway with a car, and beyond that a quiet suburban street. 
A calendar on the wall has today's date circled with "INTERVIEW - 10:00 AM" written on it.
Alex is in bed, having just woken up to the alarm. The automatic coffee maker has finished brewing in the kitchen.""",
    start_time='2024-05-15T06:45:00', scenario_module=suburban,
    extras=[Susan, Receptionist, Interviewer],
    server_name=server_name
)

Alex.add_perceptual_input("Your alarm just went off. It's 6:45 AM.", mode='internal')
Alex.add_perceptual_input("Your job interview is at 10:00 AM downtown, at the Office. It's about 30 minutes away by car.", mode='internal')
Alex.add_perceptual_input("You're still in bed, feeling groggy. You can smell coffee brewing - your automatic coffee maker started on schedule.", mode='internal')
Receptionist.mapAgent.move_to_resource('Office1')
Interviewer.mapAgent.move_to_resource('Office1')
Susan.mapAgent.x = Alex.mapAgent.x
Susan.mapAgent.y = Alex.mapAgent.y
W.reference_manager.declare_relationship('Receptionist', 'works at', 'Office1', 'employer')
W.reference_manager.declare_relationship('Interviewer', 'works at', 'Office1', 'employer')
W.reference_manager.declare_relationship('Susan', 'partner of', 'Alex', 'partner of')

