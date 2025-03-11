# Custom terrain types for suburban environment
import asyncio
from enum import Enum
import sys, os
import wave
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sim.context as context
import sim.agh as agh
import plays.config as configuration
from sim.cognitive.driveSignal import Drive

server_name = configuration.server_name
class SuburbanTerrain(Enum):
    House = 1
    Yard = 2
    Street = 3
    Sidewalk = 4
    Park = 5

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
Alex = agh.Character("Alex", """You are Alex, a 34-year-old software developer.
You live alone in a suburban house with a small yard.
You're organized but often running late in the mornings.
You have a job interview today at 9:00 AM at a company downtown.
You speak in a pragmatic, straightforward manner.
Your name is Alex.""", server_name=server_name)

Alex.drives = [
    Drive("getting to the job interview on time and making a good impression"),
    Drive("maintaining basic needs: hygiene, food, and appropriate appearance"),
    Drive("managing anxiety about the interview"),
    Drive("keeping your home in reasonable order")
]

Alex.add_perceptual_input("Your alarm just went off. It's 7:15 AM. Your interview is at 9:00 AM downtown, about 30 minutes away by car.", 'internal')
Alex.add_perceptual_input("You're still in bed, feeling groggy. You can smell coffee brewing - your automatic coffee maker started on schedule.", 'internal')

# Setting up the world context
W = context.Context([Alex],
"""A suburban house interior, early morning with soft light coming through blinds. A bedroom with rumpled sheets, an alarm clock showing 7:15 AM. A closet contains professional clothing. A bathroom with shower is adjacent. The kitchen has a coffee maker that has just finished brewing. Outside is a driveway with a car, and beyond that a quiet suburban street. A calendar on the wall has today's date circled with "INTERVIEW - 9:00 AM" written on it.
Alex is in bed, having just woken up to the alarm. The automatic coffee maker has finished brewing in the kitchen.""", 
terrain_types=SuburbanTerrain, 
resources=SuburbanResource, 
server_name=server_name)