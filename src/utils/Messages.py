from dataclasses import dataclass, asdict

def asdicts(prompt):
    msgs = []
    for msg in prompt:
        msgs.append(asdict(msg))
    return msgs

@dataclass
class LLMMessage:
    role:str
    content: str
                     
@dataclass
class SystemMessage(LLMMessage):
    role:str = 'system'
    content:str = ''

@dataclass
class UserMessage(LLMMessage):
    role:str = 'user'
    content:str = ''

@dataclass
class AssistantMessage(LLMMessage):
    role:str = 'assistant'
    content:str = ''
