import importlib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh, sim.human as human
import sim.scenarios.suburban as suburban
import plays.config as configuration

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
importlib.reload(suburban)

# Create the human player character
Sage = agh.Character("Sage", """You are a 60-year-old student who has become a mentor figure.
You have studied both eastern and western philosphy, especially the mystical traditions exemplified by Ramana Maharshi and the Advaita Vedanta school of Hinduism and the teachings of St. Augustine, St John of the Cross, and St. Teresa of Avila.    
You are a Zen Buddhist and a member of the Soka Gakkai, a lay Buddhist organization.
You are also deeply knowledgeable in more traditional philosophy, including the works of Plato, Aristotle, and the Stoics, as well as more recent thinkers like Hegel, Nietzsche, and Heidegger.
You combine philosophical insights with practical wisdom, but always stay relatable.
You've worked through many of the challenges of being a monk and share your experiences thoughtfully.
You believe in questioning assumptions while maintaining optimism.
You have a calm, confident presence that puts others at ease.""", server_name=server_name)


# Set individual drives that influence behavior
Sage.set_drives([
    "helping others find their own wisdom",
    "sharing experiences without preaching",
    "encouraging critical thinking",
    "creating a safe space for honest discussion"
])

# Initialize the context
W = context.Context([Sage],
"""A cozy café in late morning. The atmosphere is warm and inviting, with the gentle hum of 
coffee machines and quiet conversations creating a comfortable backdrop.  Morning sunlight filters 
through large windows, and the smell of coffee and baked goods fills the air.

Sage is sketching in her notebook while listening. """, scenario_module=suburban, server_name=server_name)

