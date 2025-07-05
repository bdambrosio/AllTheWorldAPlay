## ImprovAI[^1] Version 2.1
*rename will migrate to all other appearances of label, and reflects slight project refocus.*
### A playground for cognitive software and other beings.
#### minor update - installing scaffolding to prevent characters from delaying decisions indefinitely.

### long-term plan update
ImprovAI provides a stage for mixed human / AI-improv performances.
Given a short scenario, it:
- generates a cast of cognitively-deep AI actors (my primary research interest)
- Allows each character to choose how to act at the Act, Scene, Goal, Task, and individual Action levels.
- Under the overall guidance of a Director whose job it is to coordinate the actors into a coherent narrative.
- The role of Director can be flexibly assigned to the system or a human at any decision level, separately for each character.

Sample replays at [demo](https://demo.tuuyi.com)

![Play screenshot](docs/images/WebworldMain.png)

more at [Slide Show](http://www.tuuyi.com)
or [Devin] ([https://app.devin.ai/wiki/bdambrosio/AllTheWorldAPlay])

example scenario:
```
map_file_name='forest.py' # needed to trigger narrative creation

# Character definitions
S = NarrativeCharacter("Samantha", """Samantha, a healthy, dark-haired young woman in grey hiking pants and a blue pendleton shirt. 
You love the outdoors and hiking.
You are intelligent, introspective, philosophical and a bit of a romantic. 
You have a uncomfortable history, maybe it is just as well you don't remember it.
You are informal, chatty, and are a playful when relaxed, but at other times can be argumentative and defensive.
You are comfortable on long treks, and are unafraid of hard work. 
You are wary of strangers.""", 
server_name=server_name)

S.set_drives([
    "solve the mystery of how they ended up in the forest. Find a way back home.",
    "adventure, excitement, and the thrill of the unknown.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy."
])


J = NarrativeCharacter("Joe", """Joe, a healthy young man, short beard  and dark hair, in grey chinos and a red t-shirt. 
You are intelligent and self-sufficient. 
You are informal and somewhat impulsive. 
You are strong, and think you love the outdoors, but are basically a nerd.
You yearn for something more, but don't know what it is.
You are socially awkward, especially around strangers. 
You speak informally, but occasionally in a 'budding scientist' style.""",
server_name=server_name)

J.set_drives([
    "solve the mystery of how they ended up in the forest. Find a way out of the forest.",
    "safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.",
    "companionship, community, family, acceptance, trust, intimacy.",
    "immediate physiological needs: survival, shelter, water, food, rest."
])
S.add_perceptual_input("You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?", 'internal')
J.add_perceptual_input("You think Ugh. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?", 'internal')
J.add_perceptual_input("You think Whoever she is, she is pretty!", 'internal')


# Create context with forest scenario
W = context.Context([S, J],
    """A temperate, mixed forest-open landscape with no buildings, no roads, no other signs of humanity. 
    It is a early morning on what seems like it will be a warm, sunny day.
    Two people are standing in the middle of the forest, looking around in confusion.""",
    scenario_module=forest,  # Pass the forest scenario module
    server_name=server_name)
```

## A stage on which plays are presented, with AI cognitive agents as actors.

What's a cognitive agent? Well (my defs): *agents* are entities that can do things. *cognition* is using explicit representation and reasoning. So AI cognitive agents are hardware/software-based entities that build and reason over explicit representations to think and act. These do so on stage[^2].

Why? 
- My AGB (artificial general Being) test - In order for these plays to be interesting, the actors must be *interesting* for a sufficiently long period of time (longer than the typical chatbot is the first threshold). So, motivation 1 - a testbed for my ideas.
- Pychology/Voyeurism - Why do interesting cognitive simulacra do what they do? The UI allows inspection into most of the cognitive state of any actor at any time - drives, *signals* (perceptually salient indicators from sensors), emotions, tasks and plans, acts, thoughts about other actors, etc. If that isn't enough, you can chat with any actor and simply ask them.
- Design your own plays and see how the actors handle various situations.
- It's just plain fun

Limitations:
This is very much *Alpha* software. In particular, load/save only saves actor-models and hasn't been tested recently. Also, this is not clone and run. There may be hardcoded paths you will need to edit for things like LLM model files, for example (post an issue, please!). It doesn't (yet) know about your GPU config, but does has most of the major direct providers, see src/plays/config.py Finally, it is slow. About 1/3 real-time, probably, IF you use a low-latency LLM provider. Many many LLM calls per actor. So, for example, I've found DeepSeek, while attractively priced, too long latency to be useful once everyone discovered them. openai 4.1-mini or comparable competitors aren't bad, gemma-3-27b-it is fun to run local - not up to 4.1-mini dialog quality, but much edgier.

So this is NOT at this time for those unprepared to dig into the software at all the levels required to make llm-based raw python work. However, I can promise to work hard to support anyone willing to give it a try. E.G. find a hardcoded path I need to fix, post an issue!. Having said that, I'm a lone developer, and more documentation (e.g. how to write plays, and the scenarios underneath them) are high priority. However, I get distracted by new ideas, hence the new narrative capability.

More info in wiki.

## Installation:
The following install has been tested on Ubuntu 22.04 (Llambda instance with A10) or try containers!
```code
# first install python 3.12
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa   # press ↵ if prompted
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
PY312=/usr/bin/python3.12

#create venv$PY312 -m venv ~/venvs/myproj-312
source ~/venvs/myproj-312/bin/activate#clone repo and install requirementsgit clone https://github.com/bdambrosio/AllTheWorldAPlay.git
pip install faiss-gpu-cu12[fix_cuda]
cd AllTheWorldAPlay/src
python -m pip install -r requirements.txt
cd sim
export OPENAI_API_KEY="....." # add other api keys as needed, otherwise ignore errors on missing keys
uvicorn main:app --port 8000
# you should now see the python engine startup sequence, ending in: Converting message: show_update

# startup UI
cd webworld/src/
sudo apt install npm
npm install # ignore warnings
npm start
# you should now http://<host-ip>:3000 in your browser and see empty UI!

# startup image-server
cd ../../../utils # ie, to AllTheWorldAPlay/src/utils/
fastapi run lcmLora-serve.py --port 5008 &
# this should start the image server. First run will have to download the lcmLora model from huggingface
```

## Use

```
See install script, just skip the install steps!
#   don't believe 'Application startup complete', that's just the python UI handler
# now go to *host*:3000 in your browser, click Initialize (or screen refresh first if nothing happens)
# if all is well that will display a combo box of available plays. Pick one and click the load button. Wait.
# Note: at the moment the Alex, Demo, laTerre, and lost plays should work, I'm still porting the others.
```
API keys to put in your env (minimally, only the one you use, if not local):
  ```code
export OPENAI_API_KEY
export GOOGLE_KEY=
export GOOGLE_CX=
export MISTRAL_API_KEY=
export CLAUDE_API_KEY=
export DEEPSEEK_API_KEY=
export HIVE_API_KEY= (for Hive image server if not running a local tti)
export XAI_API_KEY=
export COHERE_API_KEY=
export OPENROUTER_API_KEY= (don't forget to set model name in plays/config.py)
```
- ATWAP uses several ports, including 3000, 5000, 5008, 5555, 5556, and 8000. That's probably horrible. I apologize. I am a jack of all trades, master of none.
- It needs two external services - a model (LLM) server, and a tti (text-to-image) server.
  - model server: in plays/config.py you can uncomment the appropriate line to use a variety of sources. Put your api-key in your env.
      - 'deepseeklocal' is in fact vllm, I'll explain that in a future update, post an issue if you need to know more.
      - 'local' is for running locally llama.cpp or exllamav2 that will prompt for the model to load. run ```code fastapi run exl2-server.py --port 5000 &```. I run Llama3.3-70B-Instruct locally in 8 bit exl2 on a pair of RTX6000Ada. At the time I write this Grok will give you $150/mo credits if you allow them to capture your traffic for training. Grok2 is fast and quite nice.
  - image server: I use src/utils/lcmlora-serve.py. It's fast, only uses ~ 4GB of vram. Images are mostly just eye candy, don't expect much from them. to run it cd to utils and run ```code fastapi run lcmLora-serve.py --port 5008 &```. Alternately, you can run (or adapt) ```code fastapi run hive_serve.py --port 5008 &``` or adapt the code to use your favorite image server. 

## Now what?
- as above, use Initialize to load a play
- Step allows one actor to act
- Run starts free-running round robin of actors
- Pause will stop 'Run' at the end on the current actor actions
- tabs in upper left - one for each actor
- 'Explore' button on bottom of an actor tab will open a modal for deeper exploration, including a chat tab. Be patient.
- 'Director's chair in bottom right of main screen allows you to choose actions for actors. Only Acts works at the moment, I think.
- Overall, be very patient. Sorry, they have a lot to do, it takes time.
- See [Slides](https://tuuyi.com) for more on what you should see / can do.

## Random notes
- if you don't like how a play is going, reload it! Every run is different.
- If you don't like something about a character, tweak the play (they are all in src/plays). Plays are python, best to just edit the text strings unless you've looked at the engine code.
- Play debugging - 4/11/2025 new logging of goals / tasks / actions to sim/simulation.log.
  - Load the log into utils/format_prompt.py using the Import File button, the Format Text to see the generation prompts and responses and execution trace.

Ideas / contributions (via PRs?) most welcome.

[^1]: An Improvisational AI Theater. If this term is copywritten (? is that a word) let me know, sorry.
[^2]: with a simple world sim underneath they can interact with.
