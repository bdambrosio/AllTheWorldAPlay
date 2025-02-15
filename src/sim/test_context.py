import unittest
from sim.context import Context
from sim.agh import Agh

class TestContext(unittest.TestCase):
    def setUp(self):
        # Create actors
        self.actor1 = Agh(name="Actor1", character_description="A test actor")
        self.actor2 = Agh(name="Actor2", character_description="A test actor") 
        self.actor3 = Agh(name="Actor3", character_description="A test actor")
        
        # Create context with actors
        self.context = Context(
            actors=[self.actor1, self.actor2, self.actor3],
            situation="Test situation",
            mapContext=False  # Disable map to keep test simple
        )

    labels = ['Delta', 'Omicron', 'Zeta', 'Kappa']
    #a1 priorities = ['#name Inspect Farm Crops\n#description Check wheat crop condition\n#reason Ensure food source stable\n#actors Jean\n#committed False\n#termination Crop health assessed fully ##', '#name Inspect Threshing Machine\n#description Inspect threshing machine together\n#actors Jean, Francoise\n#reason fix threshing machine issue\n#termination machine is fixed\n#committed True', "#name inspect threshing machine\n#description examine Francoise's threshing machine\n#actors Jean\n#reason to help Francoise with harvest\n#termination Threshing machine is fixed\n#committed True", "#name lend tools to Francoise\n#description give spare blade and rope\n#actors Jean\n#reason to aid Francoise's harvest\n#termination Tools are given to Francoise\n#committed True"]
    #a2 priorities = ['#name Visit Jean Macquart Field\n#description Talk to Jean Macquart briefly\n#reason Learn about his farm work\n#actors Francoise Jean Macquart\n#committed False\n#termination Conversation with Jean Macquart', '#name Finish Family Chores\n#description Help family with remaining tasks\n#reason Support family labor needs\n#actors Francoise\n#committed False\n#termination Chores are completed tonight', "#name ask pa about help\n#description ask pa about Jean's help\n#actors Francoise\n#reason to get pa's approval first\n#termination pa's approval obtained\n#committed True"]
def test_next_act_round_robin(self):
        """Test that next_act() returns actors in round-robin order when no committed tasks"""
        # First call should return first actor
        task, actors = self.context.next_act()
        self.assertEqual(actors[0], self.actor1)
        
        # Second call should return second actor
        task, actors = self.context.next_act()
        self.assertEqual(actors[0], self.actor2)
        
        # Third call should return third actor
        task, actors = self.context.next_act()
        self.assertEqual(actors[0], self.actor3)
        
        # Fourth call should wrap around to first actor
        task, actors = self.context.next_act()
        self.assertEqual(actors[0], self.actor1)

if __name__ == '__main__':
    unittest.main() 