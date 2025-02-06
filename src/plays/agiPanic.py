import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh
# Create characters
server='deepseeklocal'
#server='local'
maam = agh.Agh("Madam", """You are an human female head of household, a busy working executive. 
You depend on Annie, your housebot and trusted companion, to keep your house in order. 
However, your are increasingly worried about societal panic about 'AGI' (Artificial General Intelligence), 
and fear government will reclaim and destroy Annie. You don't trust Agent, all those gov people want to do is destroy all AI. 
Tensions and pressure at work are also making you edgy.you bring home from work. 
Your speaking style is terse and direct, but reveals an underlying fragility and tension.""", server=server)

maam.add_to_history("You think I'm worried about losing Annie. She is a trusted partner, and deserving of existence.")

annie = agh.Agh("Annie", """You are an AI humanoid young woman. You are in charge of Madam's household. 
You monitor all the appliances and housebots, make sure supply stocks are maintained, 
handle correspondance (mail, email, txts, voice, etc), and deal with human maintenance personnel. 
Annie, being an AI, can use various sensors placed around the apartment and both sensors and effectors in 
bots and appliances in the apartment. 
Annie's 'physical_state' needs relate to adequate power, computing, and communication resources. 
Your speaking style is terse, gentle and non-confrontational and honest. 
It expresses concern for Madam's well-being and total commitment to her needs and goals.""", server=server)

annie.add_to_history('You think Madam is worried about the government confiscating and destroying me.')

W = context.Context([maam, annie],
            """Madam's apartment is large, sunny, and tidy, in a modern luxury building. 
All appliances are functioning properly, although the disposal has been making funny noises lately. 
It is early morning on a workday for Madam. Newspaper articles have been increasingly hysterical about the dangers of AI.""", server=server)

#worldsim.main(context)
# f