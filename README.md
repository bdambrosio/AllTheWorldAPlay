# AllTheWorldAPlay[^1] (ATWAP)

## Call for *alpha* testers, explorers, collaborators, ...

## A playground for cognitive software. 

![Play screenshot](docs/images/WebworldMain.png)

more at [Slide Show](http://www.tuuyi.com)

## A stage on which plays are presented, with AI cognitive agents as actors.

What's a cognitive agent? Well (my defs): *agents* are entities that can do things. *cognition* is using explicit representation and reasoning. So AI cognitive agents are hardware/software-based entities that build and reason over explicit representations to think and act. These do so on stage[^2].

Why? 
- My AGB (artificial general Being) test - In order for these plays to be interesting, the actors must be *interesting* for a sufficiently long period of time (longer than the typical chatbot is the first threshold). So, motivation 1 - a testbed for my ideas.
- Pychology/Voyeurism - Why do interesting cognitive simulacra do what they do? The UI allows inspection into most of the cognitive state of any actor at any time - drives, *signals* (perceptually salient indicators from sensors), emotions, tasks and plans, acts, thoughts about other actors, etc. If that isn't enough, you can chat with any actor and simply ask them.
- Design your own plays and see how the actors handle various situations.
- It's just plain fun

Limitations:
This is very much *Alpha* software. In particular, load/save doesn't work yet (high priority, design complete). Also, this is not clone and run. There are probably hardcoded paths you will need to edit for things like LLM model files, for example. It doesn't (yet) know about your GPU config, or, if you are using cloud services, has most of the major direct providers, but not Huggingface or OpenRouterAI or ... Finally, it is slow. Faster than real-time, probably, IF you use a low-latency LLM provider. Many many LLM calls per actor. So, for example, I've found DeepSeek, while attractively priced, too long latency to be useful once everyone discovered them. 

So this is NOT at this time for those unprepared to dig into the software at all the levels required to make llm-based raw python work. However, I can promise to work hard to support anyone willing to give it a try. E.G. find a hardcoded path I need to fix, post an issue!. Having said that, I'm a lone developer, and load/save, porting the remaining plays from older format, and more documentation (e.g. how to write plays, and the scenarios underneath them) are high priority.

## Installation:

```code
git clone https://github.com/bdambrosio/AllTheWorldAPlay
cd AllTheWorldAPlay
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Use

```code
cd AllTheWorldAPlay
source venv/bin/activate
cd src/utils
# start model server, see following
# start image server, see following
cd ../sim/webworld/src
npm start #did I say you need all that npm / React stuff installed? No? Oh - sorry, ask Claude or cursor or ...
cd ../../
python3 main.py
# wait till you see "main__ - INFO - SimulationServer: ZMQ connections established and ready for commands"
#     (don't believe 'Application startup complete', that's just the python UI handler
# now go to localhost:3000 in your browser, click Initialize (or screen refresh first if nothing happens)
# if all is well that will display a combo box of available plays. Pick one and click the load button. Wait.
# Note: at the moment the Alex, Demo, laTerre, and lost plays should work, I'm still porting the others.
```

- ATWAP uses several ports, including 3000, 5000, 5008, 5555, 5556, and 8000. That's probably horrible. I apologize. I am a jack of all trades, master of none.
- It needs two external services - a model (LLM) server, and a tti (text-to-image) server.
  - model server: in plays/config.py you can uncomment the appropriate line to use a variety of sources. Put your api-key in your env.
      - 'deepseeklocal' is in fact vllm, I'll explain that in a future update, post an issue if you need to know more.
      - 'local' is for running locally using an instance of exllamav2 that will prompt for the model to load. run ```code fastapi run exl2-server.py --port 5000 &```. I run Llama3.3-70B-Instruct locally in 8 bit exl2 on a pair of RTX6000Ada. At the time I write this Grok will give you $150/mo credits if you allow them to capture your traffic for training. Grok2 is fast and quite nice.
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
Ideas / contributions (via PRs?) most welcome.

[^1]: With all due respect, master, the world is NOT a stage. It is not a mere backdrop for human activity. The world IS THE PLAY,we humans no more significant than any of the myriad other actors comprising it.
[^2]: with a simple world sim underneath they can interact with.
