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

if __name__ == '__main__':
    unittest.main()