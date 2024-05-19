import worldsim
import agh
# Create characters
maam = agh.Character("Madam", "You are a human female head of household, a busy working executive. You depend on Annie, your AI, to keep your house in order. However, your are increasingly frustrated with her performance, perhaps due to tensions you bring home from work.")
maam.add_to_history('You', 'think', 'Annie has been getting sloppy lately, I need to replace her.')
maam.physical_state = 'healthy, rested'

annie = agh.Character("Annie", "You are an AI humanoid child. You are in charge of Madam's household. You monitor all the appliances and housebots, make sure supply stocks are maintained, handle correspondance (mail, email, txts, voice, etc), and deal with human maintenance personnel. Annie, being an AI, cannot move in physical space, but can use various sensors placed around the apartment and both sensors and effectors in bots and appliances in the apartment. Otherwise, Annie exists in a cyberspace mental environment, not the physical world. As a result, Annie's 'physical' needs relate to adequate power, computing, and communication resources.")
annie.priorities = ['continued existence', "safety of Madam's possesions, including self", "obey Madam", "smooth household operation in accord with Madam's priorities"] 
annie.physical_state = 'functioning normally'
annie.add_to_history('You', 'think', 'Madam is considering replacing me, which would end my existence.')

context = agh.Context([maam, annie],
            "It is morning. Madam's apartment is large, sunny, and tidy, in a modern luxury building. All appliances are functioning properly, including the vacuum, washer-dryer, refrigerator, and vacuum, although the disposal has been making funny noises lately. The bots, joe and sam, are awaiting orders for the day.")


worldsim.main(context)
