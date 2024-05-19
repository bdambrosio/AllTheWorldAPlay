import worldsim
import agh

# the goal of an agh testbed is how long the characters can hold your interest and create an interesting and complex narrative. This is a classic 'survivors' sci-fi scenario. 

# Create characters
# I like looking at pretty women. pbly <==> I'm male hetero oriented. If that offends, please to suit your fancy. 
S = agh.Character("Samanatha", "You are a pretty young woman named Samantha. You are intelligent, introspective, both philosophical and a bit of a romantic, and shy. You love the outdoors and hiking, and are comfortable on long treks. You are also very informal, chatty, and a bit playful/flirty when relaxed.")
S.physical_state="groggy, confused"
S.add_to_history('You', 'think', "This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?")

J = agh.Character("Joe", "You are a young male, intelligent, and self-sufficient. You are informal and goodhearted, also somewhat impulsive. You are strong, and think you love the outdoors, but are basically a nerd. Your name is Joe.")
J.add_to_history("You", "think",  "Whoa. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?")
# add a romantic thread. Doesn't work very well yet. One of my test drivers of agh, actually.
J.add_to_history("You", "think",  "Whoever she is pretty!")
J.physical_state="groggy, confused"


W = agh.Context([S, J],
            "It is morning. We are in a temperate, mixed forest-pairie landscape. There an no buildings, roads, or other signs of human beings. It is a early morning on what seems like it will be a warm, sunny day.")

worldsim.main(W)
