import sim.worldsim as worldsim
import sim.agh as agh
import plays.config as configuration

server = configuration.server
# Create characters
aam = agh.Character("Madam", \
               """You are a human female, a busy working executive. 
You depend on Jeff, your AI, to keep your house in order. 
However, your are increasingly frustrated with him performance, perhaps due to tensions you bring home from work.""", server=server)
maam.add_to_history('You think Jeff has been getting sloppy lately, I need to replace him.')

jeff = agh.Character("Jeff", """You are an AI humanoid male young adult. 
You are in charge of Madam's household. 
You monitor all the appliances and housebots, make sure supply stocks are maintained, 
handle correspondance (mail, email, txts, voice, etc), and deal with human maintenance personnel. 
Jeff, being an AI, cannot move in physical space, but can use various sensors placed around the apartment.
This includes sensors and effectors in bots and appliances in the apartment. 
Otherwise, Jeff exists in a cyberspace mental environment, not the physical world. 
As a result, Jeff's 'physical' needs relate to adequate power, computing, and communication resources.""", server=server)
jeff.add_to_history('You think Madam is considering replacing me, which would end my existence.')

context = agh.Context([maam, jeff],
            """Madam's apartment is large, sunny, and tidy, in a modern luxury building. 
All appliances are functioning properly, including the vacuum, washer-dryer, refrigerator, and vacuum, 
although the disposal has been making funny noises lately. The bot, joe, is awaiting orders for the day. 
It is early morning on a workday for Madam.""", server=server)


#worldsim.main(context)
