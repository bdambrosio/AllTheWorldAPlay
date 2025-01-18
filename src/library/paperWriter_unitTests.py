#paperWrite_unitTests.py
from email import utils
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from PyQt5.QtWidgets import QApplication
import sys
from planner import Planner
from utils.xml_utils import find, findall
import chat.OwlCoT as oiv

class TestPaperWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        cls.planner = Planner( None)
        
    def setUp(self):
        self.config = {
            "length": 2000,
            "model": "llm"
        }

    def test1_analyze(self):
        """Test plan analysis"""
        try:
            plan_xml = self.planner.initialize(topic="Test Research Topic")
            self.assertIsNotNone(plan_xml)
            self.assertEqual(find('<task>',plan_xml), "Test Research Topic")
            self.assertIsNotNone(find('<dscp>',plan_xml))
            self.assertIsNotNone(find('<sbar>',plan_xml))
            self.assertIsNotNone(find('<outline>',plan_xml))
            plan_xml = self.planner.analyze(plan_xml)
            self.assertIsNotNone(plan_xml)
            self.assertIsNotNone(find('<needs>',plan_xml))
            self.assertIsNotNone(find('<background>',plan_xml))
            self.assertIsNotNone(find('<observations>',plan_xml))
        finally:
            pass
                      
                      
                      
                      
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
            outlined_plan = self.planner.outline(self.__class__.test_plan_xml)
            self.assertIsNotNone(outlined_plan)
            outline = find('<outline>',outlined_plan)
            self.assertIsNotNone(outline)
            sections = findall('<section>',outline)
            self.assertGreater(len(sections), 0)
            
        finally:
            pass

class TestWriteReport(unittest.TestCase):
    def setUp(self):
        self.config = {
            "length": 2000,
            "model": "llm"
        }
        self.sample_outline = """<outline>
            <section>
                <title>Test Section</title>
                <dscp>Test Description</dscp>
                <sections>
                    <section>
                        <title>Subsection</title>
                        <dscp>Sub Description</dscp>
                    </section>
                </sections>
            </section>
        </outline>"""
        
    def test_section_structure(self):
        """Verify handling of basic section structure"""
        outline = """<outline>
            <section>
                <title>Simple Section</title>
                <dscp>Basic description</dscp>
            </section>
        </outline>"""
        report, paper_ids = self.write_report_aux(self.config, outline, outline, 500)
        self.assertIsNotNone(report)
        self.assertGreater(len(report), 0)
        self.assertIsInstance(paper_ids, set)
        
    def test_nested_sections(self):
        """Verify handling of sections containing subsections"""
        report, paper_ids = write_report_aux(self.config, self.sample_outline, self.sample_outline, 1000)
        self.assertIsNotNone(report)
        # Report should be longer due to subsection content
        self.assertGreater(len(report), 500)
        # Should reference some papers
        self.assertGreater(len(paper_ids), 0)
        
    def test_multiple_sections(self):
        """Verify handling of multiple sibling sections"""
        outline = """<outline>
            <section>
                <title>First Section</title>
                <dscp>First description</dscp>
            </section>
            <section>
                <title>Second Section</title>
                <dscp>Second description</dscp>
            </section>
        </outline>"""
        report, paper_ids = write_report_aux(self.config, outline, outline, 1000)
        self.assertIsNotNone(report)
        # Should handle both sections
        self.assertGreater(len(report), 800)
        
    def test_section_length_distribution(self):
        """Verify proper length distribution across sections"""
        outline = """<outline>
            <section>
                <title>Main Section</title>
                <dscp>Main description</dscp>
                <sections>
                    <section>
                        <title>Sub 1</title>
                        <dscp>First subsection</dscp>
                    </section>
                    <section>
                        <title>Sub 2</title>
                        <dscp>Second subsection</dscp>
                    </section>
                </sections>
            </section>
        </outline>"""
        total_length = 1500
        report, _ = write_report_aux(self.config, outline, outline, total_length)
        self.assertIsNotNone(report)
        # Total length should be approximately what was requested
        self.assertGreater(len(report), total_length * 0.8)
        self.assertLess(len(report), total_length * 1.2)
        
    def test_depth_handling(self):
        """Verify proper handling of different nesting depths"""
        deep_outline = """<outline>
            <section>
                <title>Level 1</title>
                <dscp>Top level</dscp>
                <sections>
                    <section>
                        <title>Level 2</title>
                        <dscp>Mid level</dscp>
                        <sections>
                            <section>
                                <title>Level 3</title>
                                <dscp>Deep level</dscp>
                            </section>
                        </sections>
                    </section>
                </sections>
            </section>
        </outline>"""
        report, _ = write_report_aux(self.config, deep_outline, deep_outline, 1500)
        self.assertIsNotNone(report)
        # Should handle deep nesting
        self.assertGreater(len(report), 1000)
        
    def test_empty_sections(self):
        """Verify handling of sections with no subsections"""
        outline = """<outline>
            <section>
                <title>Empty Parent</title>
                <dscp>Parent description</dscp>
                <sections>
                </sections>
            </section>
        </outline>"""
        report, _ = write_report_aux(self.config, outline, outline, 500)
        self.assertIsNotNone(report)
        # Should still generate content for empty sections
        self.assertGreater(len(report), 200)
        
    def test_malformed_outline(self):
        """Verify proper error handling for malformed outlines"""
        bad_outlines = [
            # Missing title
            """<outline><section><dscp>No title</dscp></section></outline>""",
            # Missing dscp
            """<outline><section><title>No description</title></section></outline>""",
            # Malformed sections tag
            """<outline><section><title>Bad</title><dscp>Test</dscp><section></section></section></outline>"""
        ]
        for outline in bad_outlines:
            with self.assertRaises(Exception):
                write_report_aux(self.config, outline, outline, 500)

if __name__ == '__main__':
    unittest.main()