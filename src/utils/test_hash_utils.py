import unittest
from hash_utils import findall_forms

class TestHashUtils(unittest.TestCase):
    def test_findall_forms_with_proper_termination(self):
        input_text = """#mode Think
#action I need to get moving, what's the best outfit for this interview?
#duration 2 minutes
##

#mode Do
#action Quickly get in the shower
#duration 10 minutes
##

#mode Look
#action Check the closet for my best professional attire
#duration 3 minutes
##"""
        
        forms = findall_forms(input_text)
        self.assertEqual(len(forms), 3)
        
        # Check first form
        self.assertTrue('mode Think' in forms[0])
        self.assertTrue('duration 2 minutes' in forms[0])
        self.assertFalse('mode Think\n#mode Think' in forms[0])  # No duplicate mode
        
        # Check second form
        self.assertTrue('mode Do' in forms[1])
        self.assertTrue('duration 10 minutes' in forms[1])
        self.assertFalse('mode Do\n#mode Do' in forms[1])  # No duplicate mode
        
        # Check third form
        self.assertTrue('mode Look' in forms[2])
        self.assertTrue('duration 3 minutes' in forms[2])
        self.assertFalse('mode Look\n#mode Look' in forms[2])  # No duplicate mode

    def test_findall_forms_with_missing_termination(self):
        input_text = """#mode Think
#action I need to get moving
#duration 2 minutes
#mode Do
#action Quickly get in the shower
#duration 10 minutes
#mode Look
#action Check the closet
#duration 3 minutes"""
        
        forms = findall_forms(input_text)
        self.assertEqual(len(forms), 3)
        
        # Check form separation still works
        self.assertTrue('mode Think' in forms[0])
        self.assertTrue('duration 2 minutes' in forms[0])
        self.assertTrue('##' in forms[0])  # Should add termination
        
        self.assertTrue('mode Do' in forms[1])
        self.assertTrue('duration 10 minutes' in forms[1])
        self.assertTrue('##' in forms[1])  # Should add termination
        
        self.assertTrue('mode Look' in forms[2])
        self.assertTrue('duration 3 minutes' in forms[2])
        self.assertTrue('##' in forms[2])  # Should add termination

    def test_findall_forms_empty_input(self):
        self.assertEqual(findall_forms(""), [])
        self.assertEqual(findall_forms(None), [])

    def test_findall_forms_single_form(self):
        input_text = """#mode Think
#action test
##"""
        forms = findall_forms(input_text)
        self.assertEqual(len(forms), 1)
        self.assertTrue('mode Think' in forms[0])

    def test_findall_forms_with_blank_lines(self):
        input_text = """#mode Think
#action test

##

#mode Do
#action test2

##"""
        forms = findall_forms(input_text)
        self.assertEqual(len(forms), 2)
        self.assertTrue('mode Think' in forms[0])
        self.assertTrue('mode Do' in forms[1])

    def test_findall_forms_mixed_termination(self):
        input_text = """#mode Think
#action test
##
#mode Do
#action test2
#mode Look
#action test3
##"""
        forms = findall_forms(input_text)
        self.assertEqual(len(forms), 3)
        self.assertTrue('mode Think' in forms[0])
        self.assertTrue('mode Do' in forms[1])
        self.assertTrue('mode Look' in forms[2])

    def test_findall_forms_duplicate_tags_in_different_forms(self):
        input_text = """#mode Think
#action test1
##
#mode Think
#action test2
##"""
        forms = findall_forms(input_text)
        self.assertEqual(len(forms), 2)
        self.assertTrue('action test1' in forms[0])
        self.assertTrue('action test2' in forms[1])

if __name__ == '__main__':
    unittest.main() 