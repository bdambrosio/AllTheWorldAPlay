import worldsim
import context, agh, human
# Create characters

jill = agh.Agh("Jill", "You are")
jill.drives = [
    "self-knowledge: comfort with knowing one's place in the world."
]
jill.add_to_history('Nothing except the awareness that I perceive nothing.')

#doc = human.Human('Doc', 'A self-contained computer scientist')
context = context.Context([jill],
            "A soft glow", step='static')

worldsim.main(context)
