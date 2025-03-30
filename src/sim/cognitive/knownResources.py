from __future__ import annotations
from typing import Optional
from utils.Messages import UserMessage

class KnownResource:
    def __init__(self, resource, manager):
        """An instance of a character internalmodel of another actor"""
        self.manager = manager
        self.resource = resource

    def to_string(self):
        try:
            string = f'{self.name()}: {self.description()}'
            return string
        except Exception as e:
            print(f"Error converting resource to string: {e}")
            return ''
 
    def location(self):
        return self.resource['location']
    
    def description(self):
        return self.resource['description'] if 'description' in self.resource else ''
    
    def name(self):
        return self.resource['name']
    
    def properties(self):
        return self.resource['properties']
    
    def owner(self):
        if 'owner' in self.resource['properties']:
            return self.resource['properties']['owner']
        else:
            return None

class KnownResourceManager:
    def __init__(self, owner_agh, context):
        self.owner = owner_agh
        self.context = context
        self.known_resources = {}

    def to_string(self):
        return '\n'.join([resource.to_string() for resource in self.known_resources.values()])

    def names(self):
        return [resource.name for resource in self.known_resources.values()]
    
    def resource_models(self):
        return self.known_resources.values()
    
    def add_resource_model(self, resource_id):
        """Add a resource to agent's knowledge by resource_id"""
        resource = self.context.map.get_resource_by_id(resource_id)
        if resource and resource_id not in self.known_resources:
            try:
                self.known_resources[resource_id] = KnownResource(resource, self)
            except Exception as e:
                print(f"Error adding resource model for {resource_id}: {e}")
        return self.known_resources.get(resource_id)

    def get_resource_model(self, resource_name: str, create_if_missing: bool=False) -> Optional[KnownResource]:
        resource_name = resource_name.strip().capitalize()
        if resource_name not in self.known_resources:
            if create_if_missing:
                print(f"{self.owner.name} creating model for {resource_name}")
                self.add_resource_model(resource_name)
            else:
                    return None
        return self.known_resources[resource_name] if resource_name in self.known_resources else None

    def set_all_resources_visible(self):
        for resource in self.known_resources:
            self.known_resources[resource.name].visible = True

    def set_all_resources_invisible(self):
        for resource in self.known_resources.values():
            resource.visible = False

    def add_seen_resources(self, resources):
        for resource in resources:
            self.add_resource_model(resource['id'])

    def resolve_resource(self, reference_text):
        """
        Resolve a reference to a resource
        Args:
            reference_text: Text reference to resolve
        Returns:
            tuple: (resource, canonical_name) or (None, None) if unresolved
        """
        # Try to resolve first
        if not reference_text or reference_text == '' or reference_text.lower() == 'none':
            return None, None
        resource = self.context.map.resource_registry.get(reference_text)
        if resource:
            return resource, reference_text
        return None, None