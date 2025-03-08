import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh
import plays.config as configuration

server_name = configuration.server_name
# Create characters

jill = agh.Character("Jill", "I am a young confused woman.", server_name=server_name)
jill.set_drives([
    "self-knowledge: comfort with knowing one's place in the world.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "world knowledge: understanding the world and its workings."])
jill.add_to_history('Nothing except the awareness that I perceive nothing.')


W = context.Context([jill],
            "A soft glow of light in the room, a warm, cozy atmosphere.", server_name=server_name)

#worldsim.main(context, server_name=server_name)
