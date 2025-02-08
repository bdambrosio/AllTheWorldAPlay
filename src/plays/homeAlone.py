import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh, sim.human as human
import plays.config as configuration

server = configuration.server
# Create characters

jill = agh.Agh("Jill", "I am a young confused woman.", server=server)
jill.set_drives([
    "self-knowledge: comfort with knowing one's place in the world.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "world knowledge: understanding the world and its workings."])
jill.add_to_history('Nothing except the awareness that I perceive nothing.')


W = context.Context([jill],
            "A soft glow of light in the room, a warm, cozy atmosphere.", server=server)

#worldsim.main(context, server='local')
