import unittest
import os, json, math, time, requests, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Add parent directory to path to access existing simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.referenceManager import ReferenceManager
from utils.llm_api import LLM
from sim.context import Context
from sim.agh import Character
import sim.scenarios.forest as Forest

class TestReferenceResolution():
    def setUp(self):
        self.llm = LLM('local')
        self.ref_manager = ReferenceManager(self.llm)
        
                
        self.jean = Character("Jean", 'a peasant', "a young farmer's son")
        self.charles = Character("Charles", 'a peasant', "an elderly blacksmith")
        self.marie = Character("Marie", 'a peasant', "the village baker")
        
        # Stub context with test characters
                
        self.ref_manager.context = Context([self.jean, self.charles, self.marie], 'sunday morning in a village in France', Forest, server_name='local')
        
        # Declare some relationships
        self.ref_manager.declare_relationship("Jean", "son of", "Charles", "father of")
        self.ref_manager.declare_relationship("Marie", "friend of", "Jean", "friend of")
        
    def test_resolve_relationship_reference(self):
        """Test resolving a relationship-based reference"""
        result = self.ref_manager.resolve_reference_with_llm("Jean's father")
        print(f's/b Charles: {result}')
        
    def test_resolve_descriptive_reference(self):
        """Test resolving a description-based reference"""
        result = self.ref_manager.resolve_reference_with_llm("the elderly blacksmith")
        print(f's/b Charles: {result}')
        
    def test_resolve_combined_reference(self):
        """Test resolving a reference using both relationship and description"""
        result = self.ref_manager.resolve_reference_with_llm("the young farmer's son's friend")
        print(f's/b Marie: {result}')
        
    def test_unknown_reference(self):
        """Test handling of unknown reference"""
        result = self.ref_manager.resolve_reference_with_llm("the mysterious stranger")
        print(f's/b None: {result}')

if __name__ == '__main__':
    test = TestReferenceResolution()
    test.setUp()
    test.test_resolve_relationship_reference()
    test.test_resolve_descriptive_reference()
    test.test_resolve_combined_reference()
    test.test_unknown_reference()
