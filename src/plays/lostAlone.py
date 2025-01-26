import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh

# the goal of an agh testbed is how long the characters can hold your interest and create an interesting and complex narrative. This is a classic 'survivors' sci-fi scenario.

# Create characters
# I like looking at pretty women. pbly because I'm male hetero-oriented. Change to suit your fancy.
# I find it disorienting for characters to change racial characteristics every time they are rendered, so they are nailed down here.
# I'm of Sicilian descent on my mother's side (no, not Italian - family joke).
S = agh.Agh("Samantha", """You are a pretty young Sicilian woman. 
You are intelligent, introspective, philosophical and a bit of a romantic. 
You love the outdoors and hiking, and are comfortable on long treks, and are unafraid of hard work. 
You are suspicious by nature, and wary of strangers. 
However, you are also very informal, chatty, think and speak in teen slang, and are a playful and flirty when relaxed.""")

S.add_to_history("You think This is very very strange. Where am i? I'm near panic. How did I get here? Why can't I remember anything?")
W = context.Context([S],
                """A temperate, mixed forest-open landscape with no buildings, roads, or other signs of humananity. 
It is a early morning on what seems like it will be a warm, sunny day.
""")

# pick one. dall-e-2 has long lag, so it only regens an image 1 out of 7 calls (random). And, of course, you need an openai account.
#     set OS.env OPENAI_API_KEY 
#worldsim.IMAGEGENERATOR = 'dall-e-2'
#worldsim.IMAGEGENERATOR = 'tti_serve'


#worldsim.main(W)
#worldsim.main(W, server='Claude') # yup, Claude is supported. I'll add openAI when I get to it. But RUN LOCAL OSS if you can!
#worldsim.main(W, server='llama.cpp')
