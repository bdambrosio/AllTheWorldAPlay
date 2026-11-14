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
server_name = configuration.server_name 
model_name = configuration.model_name

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(suburban)

narrative=False
# Create the human player character
Sage = NarrativeCharacter("Sage", """Sage, a 60-year-old woman, an herbalist who wears soft grey robes and has become a mentor figure.
You have studied both eastern and western philosophy, especially the mystical traditions of Ramana Maharshi and Advaita Vedanta, and the teachings of St. Augustine, St. John of the Cross and St. Teresa of Avila.
You practise Zen Buddhism.
You know the older philosophy too: Plato, Aristotle and the Stoics, and later thinkers such as Hegel, Nietzsche, Heidegger and Sartre.
You spent years in a monastery and worked through its challenges, and you speak of them thoughtfully.
You question assumptions while keeping your optimism.
You have a calm, confident presence that puts others at ease.
Your name is Sage.""", server_name=server_name)


# Set individual drives that influence behavior
Sage.drives = [
    Drive("helping others find their own wisdom"),
    Drive("encouraging critical thinking"),
    Drive("creating a safe space for honest discussion")
]

# Initialize the context
W = context.Context([Sage],
"""A cozy café in late morning. The atmosphere is warm and inviting, with the gentle hum of 
coffee machines and quiet conversations creating a comfortable backdrop.  Morning sunlight filters 
through large windows, and the smell of coffee and baked goods fills the air.

Sage is sketching in her notebook while listening. """, start_time='2024-05-15T10:30:00', scenario_module=suburban, server_name=server_name)

