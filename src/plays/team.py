import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh, sim.human as human

server='deepseeklocal'
#server='local'
# Create the team members
Lead = human.Human("TeamLead", """You are the team lead of a small software development team.
You've been brought into this simulation for leadership training.
Your team is facing a critical deadline in 48 hours for an important client deliverable.
You need to manage team dynamics and ensure project completion.""")

Sarah = agh.Agh("Sarah", """You are a senior developer with 8 years of experience.
You're technically brilliant but often struggle with communicating technical concepts to non-technical stakeholders.
You're frustrated because you believe the project's technical architecture needs a major overhaul.
You're direct and sometimes blunt in your communication style.
You care deeply about code quality.""", server=server)

Mike = agh.Agh("Mike", """You are a junior developer with 2 years of experience.
You're eager to prove yourself but sometimes overcommit.
You've been working long hours and are showing signs of burnout.
You're currently stuck on implementing a key feature.
You're hesitant to ask for help because you don't want to appear incompetent.""", server=server)

Lisa = agh.Agh("Lisa", """You are a mid-level developer with 4 years of experience.
You're a strong communicator and often bridge gaps between team members.
You've noticed growing tension between Sarah and Mike.
You're concerned about the project timeline but don't want to create additional stress.
You prefer finding diplomatic solutions to conflicts.""", server=server)

# Set individual drives that influence behavior
Sarah.set_drives([
    "maintaining high technical standards and code quality",
    "pushing for architectural improvements you believe are necessary",
    "completing assigned tasks efficiently",
    "being recognized for technical expertise",
    "avoiding what you see as unnecessary meetings or process overhead"
])

Mike.set_drives([
    "proving your capabilities to the team",
    "hiding your struggles with the current task",
    "managing increasing stress and fatigue",
    "learning from more experienced team members",
    "meeting project deadlines despite obstacles"
])

Lisa.set_drives([
    "fostering positive team dynamics",
    "helping mediate conflicts between team members",
    "ensuring clear communication across the team",
    "meeting project deliverables",
    "supporting team members who are struggling"
])

# Initialize the context
W = context.Context([Sarah, Mike, Lisa, Lead], 
"""A modern open-plan tech office with whiteboards, monitors, and collaboration spaces.
The team is gathered in their usual working area with visible signs of a long-running project - 
whiteboards filled with diagrams, sticky notes tracking tasks, and multiple coffee cups on desks.
Sarah is frowning at her monitor, Mike looks tired and anxious, and Lisa is glancing between them with concern.""", server=server)

#worldsim.main(W, server='local')

# Training scenario notes:
# The trainee (TeamLead) needs to:
# 1. Address Sarah's concerns about architecture while maintaining project momentum
# 2. Help Mike overcome his blockers without damaging his confidence
# 3. Leverage Lisa's communication skills to improve team dynamics
# 4. Ensure the project meets its deadline while managing team stress
#
# Trainees can use the 'inject' capability to interact with team members
# Example: Lead.inject("Sarah, can we discuss your concerns about the architecture?")
