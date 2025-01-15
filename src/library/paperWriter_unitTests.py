#paperWrite_unitTests.py
from email import utils
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from PyQt5.QtWidgets import QApplication
import sys
from paperWriter import PWUI
from planner import Planner
from utils.xml_utils import find, findall
import chat.OwlCoT as oiv

class TestPaperWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        cls.planner = Planner(None)
        cls.planner.wm.load('Planner.wm')
        
    def setUp(self):
        self.config = {
            "length": 2000,
            "model": "llm"
        }

    def test4_select_plan(self):
        """Test plan selection"""
        
        try:
            plan_xml = self.planner.select_plan()
            self.assertIsNotNone(plan_xml)
            self.assertIsNotNone(find('<task>',plan_xml))
            self.assertIsNotNone(find('<dscp>',plan_xml))
            self.assertIsNotNone(find('<sbar>',plan_xml))
            self.assertIsNotNone(find('<outline>',plan_xml))
            
            # Store for next test
            self.__class__.test_plan_xml = plan_xml
            
        finally:
            pass
    def test_1_initialize(self):
        """Test plan initialization"""
        
        try:
            plan_xml = self.planner.initialize(topic="Test Research Topic")
            self.assertIsNotNone(plan_xml)
            self.assertEqual(find('<task>',plan_xml), "Test Research Topic")
            self.assertIsNotNone(find('<dscp>',plan_xml))
            self.assertIsNotNone(find('<sbar>',plan_xml))
            self.assertIsNotNone(find('<outline>',plan_xml))
            
            # Store for next test
            self.__class__.test_plan_xml = plan_xml
            
        finally:
            pass

    def test_2_analyze(self):
        """Test plan analysis"""
                
        try:
            analyzed_plan = self.planner.analyze(self.__class__.test_plan_xml)
            self.assertIsNotNone(analyzed_plan)
            sbar = find('<sbar>',analyzed_plan)
            self.assertIsNotNone(sbar)
            self.assertIsNotNone(find('<needs>',sbar))
            self.assertIsNotNone(find('<background>',sbar))
            self.assertIsNotNone(find('<observations>',sbar))
            
            # Store for next test
            self.__class__.test_plan_xml = analyzed_plan
            
        finally:
            pass

    def test_3_outline(self):
        """Test outline generation"""
       
        try:
            outlined_plan = self.planner.outline(self.config, self.__class__.test_plan_xml)
            self.assertIsNotNone(outlined_plan)
            outline = find('<outline>',outlined_plan)
            self.assertIsNotNone(outline)
            sections = findall('<section>',outline)
            self.assertGreater(len(sections), 0)
            
        finally:
            pass
if __name__ == '__main__':
    unittest.main()