# AllTheWorldAPlay (ATWAP)

# A simple toy to test the Humanity of your LLM

## Beta 1.0 - Actually got this to run on a clean install on a different machine. Enjoy!

![Play screenshot](images/Play.jpg)

script a simple scenario, and watch it play out. 
I've only put two days into this so far.
- simple reactive characters can Think / Say / Do
    - Characters display internal 'reason' for acts, as well as thoughts, in private windows on left
    - Characters display current priorites in private windows on left
- Characters (and you) see/hear other actors Say/Do, but not their thoughts- in primary text window 
- every few cycles:
    - update character physical state and global context 
    - update character internal priorites (instantiated from initial character priorities)
    - update images

## Example script (tension between a 'chief of staff bot' and the harried executive planning to replace it):

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

## Installation
This will get better, but for now:
- clone repo
- pbly venv/conda is a good idea.
- in utils, run exl_server.py[^1]
    - you will need to make minor path changes
    - you will need exllamav2 and transformers installed (uses transformers to run chat templates)
- in utils, run tti_serve.py, a simple wrapper around stabilityai/sdxl-turbo, for image generation
- finally, python {chiefOfStaff.py, lost.py, myscenario.py, ...} from <localrepo>/src directory[^2]. 

Ideas / contributions (via PRs?) most welcome.

[^1]: a simple wrapper around exllamav2. Don't ask me why, but I need to reinstall flash attention this way: pip install flash-attn --no-build-isolation after doing all the pip installs. I hate python packaging. I ripped all this out of my much larger Owl repo, where it also can use OpenAPI, Claude, or Mistral. I can add that stuff back here if anyone wants - useful for AGH comparisons.

[^2]: Yeah, I know, no requirements.txt or other installer? Hey, this is <really> early, sorry. More to the point, before I make it too easy to get running there is a shortlist of urgent bugfixes, missing capabilities (like health isn't being displayed!) and improvements (esp. in planning, action determination, ....) I need to make.

