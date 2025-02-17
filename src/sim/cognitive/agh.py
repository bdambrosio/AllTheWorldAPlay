from sim.engine_service import publish

def push_state_update(self):
    """Push this Agh's state update directly via ZMQ"""
    publish({
        'type': 'character_update',
        'name': self.name,
        'data': self.get_explorer_state()
    }) 