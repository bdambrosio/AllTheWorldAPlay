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

narrative=False
# Create the human player character
Sage = NarrativeCharacter("Sage", """Sage, a 60-year-old female herbalist who wears soft grey robes and has become a mentor figure.
I have studied both eastern and western philosphy, especially the mystical traditions exemplified by Ramana Maharshi, the Advaita Vedanta school of Hinduism and the teachings of St. Augustine, St John of the Cross, and St. Teresa of Avila.    
I am a Zen Buddhist and a member of the Soka Gakkai, a lay Buddhist organization.
I am also deeply knowledgeable in more traditional philosophy, including the works of Plato, Aristotle, and the Stoics, as well as more recent thinkers like Hegel, Nietzsche, Heidegger, and Sartre.
I have worked through many of the challenges of being a monk and share your experiences thoughtfully.
I question assumptions while maintaining optimism.
I maintain a calm, confident presence that puts others at ease.""", server_name=server_name)


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

Sage is sketching in her notebook while listening. """, scenario_module=suburban, server_name=server_name)

