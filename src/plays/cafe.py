import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
from plays.scenarios import coastal

importlib.reload(configuration)  # force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(coastal)

map_file_name = 'coastal.py'

# Main character - a talented artist who finds inspiration in the coastal town
Maya = NarrativeCharacter("Maya", """You are Maya, a talented 32-year-old female watercolor artist.
You live in a small coastal town where you have your studio near the harbor.
You're warm, thoughtful, and value deep connections with others.
Your morning ritual includes a visit to Harbor Light Cafe for coffee before heading to your studio.
You sell your paintings to tourists and through the local gallery.
You enjoy the peaceful rhythm of the town, which gives you space to create, but sometimes wonder if your art could reach more people elsewhere.
Your name is Maya.""", server_name=server_name)

Maya.drives = [
    Drive("create meaningful art that captures the beauty of coastal life"),
    Drive("maintain a healthy daily routine with proper meals and breaks"),
    Drive("nurture relationships with community members who support your art"),
    Drive("find balance between creative solitude and social connection")
]

Maya.add_perceptual_input("""The morning light through your studio window is perfect for painting today, but first, you need your morning coffee ritual at Harbor Light Cafe.""", 'internal')

# Supporting character with his own craft and routines
Elijah = NarrativeCharacter("Elijah", """You are Elijah, a 35-year-old male boat builder.
You have deep roots in this small coastal town.
You're steady, reliable, and deeply connected to the natural rhythms of this place.
You have a workshop at the harbor where you build and repair wooden boats.
You take breaks for lunch at Harbor Light Cafe almost every day to break up your workday.
You value craftsmanship, tradition, and community.
You've become very closewith Maya and enjoy your conversations about artistry and craft.
You speak your mind with simple directness, occasionally with a bit of dry humor.
Your name is Elijah.""", server_name=server_name)

Elijah.drives = [
    Drive("hone your craft of traditional boat building and pass on your knowledge"),
    Drive("maintain physical strength and energy with regular meals and routines"),
    Drive("support local businesses that form the backbone of the town"),
    Drive("build meaningful connections with others who appreciate craftsmanship")
]

Elijah.add_perceptual_input("""It's almost noon, and you're getting hungry after a morning of sanding a hull. Time to head to Harbor Light Cafe for lunch, where they know exactly how you like your sandwich.""", 'internal')

# Cafe owner with her own goals and place in the community
Chrys = NarrativeCharacter("Chrys", """You are Chrys, a 38-year-old female owner of Harbor Light Cafe.
You spend most of your time at the cafe in your role as owner and primary cook and bottle-washer, but you also have a small apartment above the cafe.
Your cafe is a central gathering place in this small coastal town.
You are warm, observant, and have a talent for remembering everyone's preferences.
You left city life behind five years ago to open this cafe, and you've never regretted it.
You take pride in sourcing local ingredients and creating a welcoming atmosphere.
You've watched relationships form and community bonds strengthen across your tables.
You speak with a friendly, casual, chatty tone that puts people at ease.""", server_name=server_name)

Chrys.drives = [
    Drive("create a welcoming space that brings the community together"),
    Drive("run a sustainable business that supports local producers"),
    Drive("maintain meaningful connections with regular customers"),
    Drive("find small ways to improve people's days through food and conversation")
]

Chrys.add_perceptual_input("""Morning rush is starting to slow down. Maya should be arriving soon for her usual coffee, and Elijah will probably come by for lunch around noon. Better make sure their usual orders are ready to go.""", 'internal')

# Setting up the world context
W = context.Context([Maya, Elijah, Chrys],
"""A charming small coastal town in the morning light. Colorful buildings line the waterfront, with artists' workshops and fishing boats creating a scene of rustic charm. Harbor Light Cafe sits in a prime location overlooking the water, with outdoor seating and large windows that capture the view.

The cafe is bustling with morning activity. The smell of fresh coffee and baked goods fills the air. A few tables are occupied by locals starting their day. Maya's art studio is visible down the street, and Elijah's boat workshop can be seen at the harbor nearby.

It's 8:30 am on a clear Tuesday morning in May. The town is coming alive as shops open and fishing boats return with their morning catch.""", 
scenario_module=coastal,
server_name=server_name)

# Position characters at appropriate locations
Maya.mapAgent.move_to_resource('Studio')  # Maya at her studio gallery
Elijah.mapAgent.move_to_resource('Workshop')  # Elijah at his boat workshop
Chrys.mapAgent.move_to_resource('Cafe')  

# Let the scenario unfold naturally from here