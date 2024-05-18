# AllTheWorldAPlay (ATWAP)

A simple toy to test the Humanity of your LLM

![Play screenshot](images/Play.jpg)

script a simple scenario, and watch it play out. 
I've only put two days into this so far.
- simple reactive characters can Think / Say / Do
- - Characters display internal 'reason' for acts, as well as thoughts, in private windows on left
- - Characters display current priorites in private windows on left
- actors (and you) see/hear other actors Say/Do, but not their thoughts- in primary text window 
- every few cycles:
            - update character physical state and global context 
            - update character internal priorites (instantiated from initial character priorities)
            - update images

Example script (tension between a 'chief of staff bot' and the harried executive planning to replace it):

```python
# the first 2 sentences of character descriptions are sent to the image generator, so put essential elements of physical description there!
maam = agi.Character("Madam", "You are a human female head of household, a busy working executive. You depend on Annie, your AI, to keep your house in order. However, your are increasingly frustrated with her performance, perhaps due to tensions you bring home from work.")
# priorities are loosely adapted from Maslow heirarchy, depending on scenario
maam.priorities=['survival', 'safety, acceptance, community']
maam.physical_state = ['healthy, rested']

#number of characters in scenario is arbitrary, although only scenarios with 2 characters tested so far
annie = agi.Character("Annie", "You are an AI agent in charge of Madam's household. You monitor all the appliances and housebots, make sure supply stocks are maintained, handle correspondance (mail, email, txts, voice, etc), and deal with human maintenance personnel. Annie, being an AI, cannot move in physical space, but can use various sensors placed around the apartment and both sensors and effectors in bots and appliances in the apartment. Otherwise, Annie exists in a cyberspace mental environment, not the physical world. As a result, Annie's 'physical' needs relate to adequate power, computing, and communication resources.")
annie.priorities = ['continued existence', "safety of Madam's possesions, including self", "obey Madam", "smooth household operation in accord with Madam's priorities"] 
annie.physical_state = ['functioning normally']

# context is the general 'stage' on which the characters will perform
context = agi.Context([maam, annie],
            "It is morning. Madam's apartment is large, sunny, and tidy, in a modern luxury building. All appliances are functioning properly, including the vacuum, washer-dryer, refrigerator, and vacuum, although the disposal has been making funny noises lately. The bots, joe and sam, are awaiting orders for the day.")

worldsim.main(context)
```
