#paperWrite_unitTests.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from PyQt5.QtWidgets import QApplication
import sys
from library.paperWriter import (
    plan_search, s2_search, write_report_aux,
    plans, save_plans
)
import json
import os

class TestPaperWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        
    def setUp(self):
        # Basic test configuration
        self.config = {
            "Write": {"exec": "Yes", "model": "llm"},
            "Search": {"exec": "No", "model": "llm"},
            "Query": {"exec": "No", "model": "llm"},
            "SBAR": {"exec": "No", "model": "llm"},
            "Outline": {"exec": "No", "model": "llm"},
            "ReWrite": {"exec": "No", "model": "llm"}
        }
        
        # Basic test outline
        self.test_outline = {
            "title": "Test Paper",
            "dscp": "A test description",
            "length": 400,
            "rewrites": 1,
            "sections": [
                {
                    "title": "Section 1",
                    "dscp": "First section",
                    "length": 200
                },
                {
                    "title": "Section 2",
                    "dscp": "Second section",
                    "length": 200
                }
            ]
        }

    def test_plan_persistence(self):
        """Test plan saving and loading"""
        test_plan = {
            "name": "test_plan",
            "dscp": "test description",
            "task": "test task",
            "outline": self.test_outline
        }
        
        # Save plan
        plans["test_plan"] = test_plan
        save_plans()
        
        # Verify file exists and contains our plan
        self.assertTrue(os.path.exists("./arxiv/arxiv_plans.json"))
        with open("./arxiv/arxiv_plans.json", 'r') as f:
            loaded_plans = json.load(f)
        self.assertIn("test_plan", loaded_plans)
        
        
    def test_write_report_aux_basic(self):
        """Test basic report writing without resources"""
        section, paper_ids = write_report_aux(
            config=self.config,
            paper_outline=self.test_outline,
            section_outline=self.test_outline["sections"][0],
            length=200
        )
        self.assertIsNotNone(section)
        self.assertIsInstance(paper_ids, set)
        
    def test_write_report_aux_with_resources(self):
        """Test report writing with provided resources"""
        # Mock resources - would need actual S2 paper IDs and texts
        mock_resources = ([123], [["test section"]])
        section, paper_ids = write_report_aux(
            config=self.config,
            paper_outline=self.test_outline,
            section_outline=self.test_outline["sections"][0],
            length=200,
            resources=mock_resources
        )
        self.assertIsNotNone(section)
        self.assertIsInstance(paper_ids, set)
        
    def test_s2_search_basic(self):
        """Test basic S2 search functionality"""
        result = s2_search(
            self.config,
            self.test_outline,
            self.test_outline["sections"][0]
        )
        # s2_search modifies local library, no return value
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()