import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.interpreter import Interpreter, InvalidAction
from chat.OwlCoT import OwlInnerVoice


class TestInterpreter(unittest.TestCase):
    def setUp(self):
        #self.cot = OwlInnerVoice()
        self.interpreter = Interpreter()

    def test_basic_assign(self):
        """Test basic variable assignment"""
        action = {
            "label": "test1",
            "action": "assign", 
            "arguments": "test_value", 
            "result": "$test_var"
        }
        self.interpreter.do_item(action)
        self.assertTrue(self.interpreter.wm.has("$test_var"))
        self.assertEqual(self.interpreter.wm.get("$test_var")["item"], "test_value")

    def test_variable_substitution(self):
        """Test variable substitution in strings"""
        steps = [
            {"label": "assign1", "action": "assign", "arguments": "world", "result": "$var1"},
            {"label": "tell1", "action": "tell", "arguments": "hello $var1", "result": "$Trash"}
        ]
        self.interpreter.interpret(steps)
        self.assertTrue(self.interpreter.wm.has("$var1"))
        self.assertEqual(self.interpreter.wm.get("$var1")["item"], "world")

    def test_invalid_action(self):
        """Test handling of invalid action"""
        action = {
            "label": "invalid1",
            "action": "invalid_action",
            "arguments": "test",
            "result": "$test_var"
        }
        result = self.interpreter.do_item(action)
        self.assertEqual(result, "action not yet implemented")

    def test_llm_basic(self):
        """Test basic LLM query"""
        action = {
            "label": "llm1",
            "action": "llm",
            "arguments": "What is 2+2?",
            "result": "$math_result"
        }
        self.interpreter.do_item(action)
        self.assertTrue(self.interpreter.wm.has("$math_result"))
        result = self.interpreter.wm.get("$math_result")["item"]
        self.assertIn("4", result)  # The answer should contain "4"

    def test_assign_and_tell(self):
        """Test assigning a value and telling it"""
        steps = [
            {"label": "assign1", "action": "assign", "arguments": "Hello World", "result": "$greeting"},
            {"label": "tell1", "action": "tell", "arguments": "$greeting", "result": "$Trash"}
        ]
        self.interpreter.interpret(steps)
        self.assertTrue(self.interpreter.wm.has("$greeting"))
        self.assertEqual(self.interpreter.wm.get("$greeting")["item"], "Hello World")

    def test_extract(self):
        """Test content extraction"""
        steps = [
            {"label": "assign1", 
             "action": "assign", 
             "arguments": "The sky is blue and grass is green", 
             "result": "$text"},
            {"label": "extract1", 
             "action": "extract", 
             "arguments": ("color of sky", "$text"), 
             "result": "$sky_color"},
            {"label": "tell1", 
             "action": "tell", 
             "arguments": ("$sky_color"), 
             "result": "$text"}
        ]
        self.interpreter.interpret(steps)
        self.assertTrue(self.interpreter.wm.has("$sky_color"))
        result = self.interpreter.wm.get("$sky_color")["item"].lower()
        self.assertIn("blue", result)

    def test_memory_persistence(self):
        """Test that working memory persists between operations"""
        # First operation with clean memory
        interpreter1 = Interpreter(reload_memory=False)
        step1 = {
            "label": "store1",
            "action": "assign",
            "arguments": "test data",
            "result": "$persistent"
        }
        interpreter1.do_item(step1)
        
        # Verify immediate storage
        self.assertTrue(interpreter1.wm.has("$persistent"))
        initial_value = interpreter1.wm.get("$persistent")["item"]
        self.assertEqual(initial_value, "test data")
        
        # Force persistence and close
        interpreter1.persist_memory()
        interpreter1.close()
        
        # Create new interpreter instance with reload
        interpreter2 = Interpreter(reload_memory=True)
        
        # Verify data persisted
        self.assertTrue(interpreter2.wm.has("$persistent"))
        loaded_value = interpreter2.wm.get("$persistent")["item"]
        self.assertEqual(loaded_value, "test data")
        
        interpreter2.close()

    def test_if_condition_true(self):
        """Test if-condition when condition is true"""
        steps = [
            {"label": "assign1", "action": "assign", "arguments": "5", "result": "$value"},
            {"label": "if1", "action": "if", 
             "arguments": ("$value > 3", [
                 {"action": "assign", "arguments": "true", "result": "$result"}
             ]),
             "result": "$if_result"}
        ]
        self.interpreter.interpret(steps)
        self.assertTrue(self.interpreter.wm.has("$result"))
        self.assertEqual(self.interpreter.wm.get("$result")["item"], "true")

    def test_if_condition_false(self):
        """Test if-condition when condition is false"""
        steps = [
            {"label": "assign1", "action": "assign", "arguments": "2", "result": "$value"},
            {"label": "if1", "action": "if", 
             "arguments": ("$value > 3", [
                 {"action": "assign", "arguments": "true", "result": "$result"}
             ]),
             "result": "$if_result"}
        ]
        self.interpreter.interpret(steps)
        self.assertFalse(self.interpreter.wm.has("$result"))

    def test_while_loop_counter(self):
        """Test while-loop with counter"""
        steps = [
            {"label": "assign1", "action": "assign", "arguments": "3", "result": "$counter"},
            {"label": "assign2", "action": "assign", "arguments": "0", "result": "$sum"},
            {"label": "while1", "action": "while",
             "arguments": ("$counter > 0", [
                 {"action": "assign", "arguments": "$sum + $counter", "result": "$sum"},
                 {"action": "assign", "arguments": "$counter - 1", "result": "$counter"}
             ]),
             "result": "$while_result"}
        ]
        self.interpreter.interpret(steps)
        self.assertTrue(self.interpreter.wm.has("$counter"))
        self.assertTrue(self.interpreter.wm.has("$sum"))
        self.assertEqual(self.interpreter.wm.get("$counter")["item"], "0")
        self.assertEqual(self.interpreter.wm.get("$sum")["item"], "6")  # 3 + 2 + 1

    def test_while_loop_early_exit(self):
        """Test while-loop with break condition"""
        steps = [
            {"label": "assign1", "action": "assign", "arguments": "5", "result": "$counter"},
            {"label": "while1", "action": "while",
             "arguments": ("$counter > 0", [
                 {"action": "if", 
                  "arguments": ("$counter == 3", [
                      {"action": "break", "arguments": None, "result": None}
                  ]),
                  "result": "$if_result"},
                 {"action": "assign", "arguments": "$counter - 1", "result": "$counter"}
             ]),
             "result": "$while_result"}
        ]
        self.interpreter.interpret(steps)
        self.assertTrue(self.interpreter.wm.has("$counter"))
        self.assertEqual(self.interpreter.wm.get("$counter")["item"], "3")  # Should stop at 3

if __name__ == '__main__':
    unittest.main()