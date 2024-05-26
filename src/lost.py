import worldsim
import agh
import llm_api
# the goal of an agh testbed is how long the characters can hold your interest and create an interesting and complex narrative. This is a classic 'survivors' sci-fi scenario. 

# Create characters
# I like looking at pretty women. pbly because I'm male hetero oriented. If that offends, please change to suit your fancy.
# I find it disorienting for characters to change racial characteristics every time they are rendered, so they are nailed down here.
# I'm of Sicilian descent on my mother's side (no, not Italian - family joke).
S = agh.Agh("Samantha", """You are a pretty young Sicilian woman. 
You love the outdoors and hiking.
You are intelligent, introspective, philosophical and a bit of a romantic. 
You are comfortable on long treks, and are unafraid of hard work. 
You are suspicious by nature, and wary of strangers. 
However, you are also very informal, chatty, think and speak in teen slang, and are a playful and flirty when relaxed. 
Your name is Samanatha""")

S.update_physical_state('MentalState', '<MentalState>groggy and confused</MentalState>')
S.update_physical_state('Fear', '<Fear>High</Fear>')
S.add_to_history('You', 'think', "This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?")

J = agh.Agh("Joe", """You are a young Sicilian male, intelligent, and self-sufficient. You are informal and somewhat impulsive. 
You are strong, and think you love the outdoors, but are basically a nerd.
You are socially awkward, especially around strangers. Your name is Joe.""")

J.add_to_history("You", "think",  "Ugh. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?")
# add a romantic thread. Doesn't work very well yet. One of my test drivers of agh, actually.
J.add_to_history("You", "think",  "Whoever she is, she is pretty!")
J.update_physical_state('Hunger', "<Hunger>High</Hunger>")
J.update_physical_state('Fear', '<Fear>Medium</Fear>')
J.update_physical_state('MentalState', "<MentalState>Surprised</MentalState>")


# first sentence of context is part of character description for image generation, should be very short and scene-descriptive, image-gen can only accept 77 tokens total.
W = agh.Context([S, J],
                """A temperate, mixed forest-open landscape with no buildings, roads, or other signs of humananity. It is a early morning on what seems like it will be a warm, sunny day.
""")

# pick one. dall-e-2 has long lag, so it only regens an image 1 out of 7 calls (random). And, of course, you need an openai account.
#     set OS.env OPENAI_API_KEY 
#worldsim.IMAGEGENERATOR = 'dall-e-2'
worldsim.IMAGEGENERATOR = 'tti_serve'

worldsim.main(W)
#worldsim.main(W, server='Claude') # yup, Claude is supported. I'll add openAI when I get to it. But RUN LOCAL OSS if you can!
#worldsim.main(W, server='llama.cpp')
