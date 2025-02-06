import hash_utils

print(hash_utils.find('name', '#name John Doe\n##'))  
print('****')
print(hash_utils.set('name', '#person\n#name John Doe\n#description A software engineer\n#age 30##', 'Jane Smith'))
print('****')
text = """
#person
#name John Doe
#description A software engineer
#age 30
##"""

print(hash_utils.find('name', text))
print('****')
print(hash_utils.set('name', text, 'Jane Smith'))
print('****')

text = """#plan
#name Assess Surroundings
#description Check environment for safety
#reason Ensure immediate physical safety
#termination Feel safe and aware

#plan
#name Gather Food
#description Collect berries and other resources
#reason Need stable food source soon
#termination Have enough food for day

#plan
#name Approach Agent Joe
#description Cautiously talk to nearby agent
#reason Potential ally or information source
#termination Get helpful info or reject
"""

print(hash_utils.findall('plan', text))
print('****')

