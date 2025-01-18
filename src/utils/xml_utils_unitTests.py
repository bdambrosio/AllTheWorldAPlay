import unittest
from xml_utils import format_xml, XMLFormatError, findall

class TestXMLUtils(unittest.TestCase):
    def test_simple_tag(self):
        """Test formatting of single tag"""
        xml = "<tag>value</tag>"
        expected = "<tag>value</tag>"
        self.assertEqual(format_xml(xml), expected)

    def test_nested_tags(self):
        """Test formatting of nested tags"""
        xml = "<outer><inner>value</inner></outer>"
        expected = """<outer>
  <inner>value</inner>
</outer>"""
        self.assertEqual(format_xml(xml), expected)

    def test_multiple_nested(self):
        """Test multiple levels of nesting"""
        xml = "<l1><l2><l3>value</l3></l2></l1>"
        expected = """<l1>
  <l2>
    <l3>value</l3>
  </l2>
</l1>"""
        self.assertEqual(format_xml(xml), expected)

    def test_siblings(self):
        """Test sibling tags at same level"""
        xml = "<parent><child1>v1</child1><child2>v2</child2></parent>"
        expected = """<parent>
  <child1>v1</child1>
  <child2>v2</child2>
</parent>"""
        self.assertEqual(format_xml(xml), expected)

    def test_error_handling(self):
        """Test error conditions"""
        with self.assertRaises(XMLFormatError):
            format_xml(None)  # None input
            
        with self.assertRaises(XMLFormatError):
            format_xml("")    # Empty input
            
        with self.assertRaises(XMLFormatError):
            format_xml("<tag>value")  # Unclosed tag
            
        with self.assertRaises(XMLFormatError):
            format_xml("<>value</>")  # Empty tag name

    def test_processing_instruction(self):
        """Test handling of processing instructions"""
        xml = "<?xml version='1.0'?><tag>value</tag>"
        expected = "<tag>value</tag>"
        self.assertEqual(format_xml(xml), expected)

    def test_repeated_tag_names(self):
        """Test handling of repeated tag names in hierarchy"""
        xml = """<section>
            <title>Main</title>
            <sections>
                <section>
                    <title>Sub</title>
                </section>
            </sections>
        </section>"""
        expected = """<section>
  <title>Main</title>
  <sections>
    <section>
      <title>Sub</title>
    </section>
  </sections>
</section>"""
        self.assertEqual(format_xml(xml.strip()), expected)

    def test_multiple_nested_sections(self):
        """Test multiple nested sections with same tag names"""
        xml = """<outline>
            <section>
                <title>First</title>
                <sections>
                    <section>
                        <title>First.1</title>
                    </section>
                    <section>
                        <title>First.2</title>
                    </section>
                </sections>
            </section>
            <section>
                <title>Second</title>
                <sections>
                    <section>
                        <title>Second.1</title>
                    </section>
                </sections>
            </section>
        </outline>"""
        expected = """<outline>
  <section>
    <title>First</title>
    <sections>
      <section>
        <title>First.1</title>
      </section>
      <section>
        <title>First.2</title>
      </section>
    </sections>
  </section>
  <section>
    <title>Second</title>
    <sections>
      <section>
        <title>Second.1</title>
      </section>
    </sections>
  </section>
</outline>"""
        self.assertEqual(format_xml(xml.strip()), expected)

    def test_deep_nesting(self):
        """Test deeply nested structure with repeated tags"""
        xml = """<section>
            <title>Level 1</title>
            <sections>
                <section>
                    <title>Level 2</title>
                    <sections>
                        <section>
                            <title>Level 3</title>
                            <sections>
                                <section>
                                    <title>Level 4</title>
                                </section>
                            </sections>
                        </section>
                    </sections>
                </section>
            </sections>
        </section>"""
        expected = """<section>
  <title>Level 1</title>
  <sections>
    <section>
      <title>Level 2</title>
      <sections>
        <section>
          <title>Level 3</title>
          <sections>
            <section>
              <title>Level 4</title>
            </section>
          </sections>
        </section>
      </sections>
    </section>
  </sections>
</section>"""
        self.assertEqual(format_xml(xml.strip()), expected)

    def test_findall_levels(self):
        """Test findall returns only matches at specified level"""
        xml = """<root>
            <section>
                <title>First</title>
                <sections>
                    <section>
                        <title>Nested</title>
                    </section>
                </sections>
            </section>
            <section>
                <title>Second</title>
            </section>
        </root>"""
        
        # Should find only top-level sections
        results = findall('<section>', xml)
        self.assertEqual(len(results), 2)
        self.assertIn('<title>First</title>', results[0])
        self.assertIn('<title>Second</title>', results[1])
        
        # Should find nested section using path
        results = findall('<section>/<sections>/<section>', xml)
        self.assertEqual(len(results), 1)
        self.assertIn('<title>Nested</title>', results[0])

    def test_findall_multiple_levels(self):
        """Test findall with deeply nested structures"""
        xml = """<outline>
            <section>
                <title>A</title>
                <sections>
                    <section>
                        <title>A.1</title>
                        <sections>
                            <section>
                                <title>A.1.1</title>
                            </section>
                        </sections>
                    </section>
                </sections>
            </section>
            <section>
                <title>B</title>
            </section>
        </outline>"""
        
        # Top level sections
        results = findall('<section>', xml)
        self.assertEqual(len(results), 2)
        self.assertIn('<title>A</title>', results[0])
        self.assertIn('<title>B</title>', results[1])
        
        # Second level sections
        results = findall('<section>/<sections>/<section>', xml)
        self.assertEqual(len(results), 1)
        self.assertIn('<title>A.1</title>', results[0])
        
        # Third level sections
        results = findall('<section>/<sections>/<section>/<sections>/<section>', xml)
        self.assertEqual(len(results), 1)
        self.assertIn('<title>A.1.1</title>', results[0])

    def test_findall_siblings(self):
        """Test findall with multiple siblings at same level"""
        xml = """<root>
            <parent>
                <child>First</child>
                <child>Second</child>
                <child>Third</child>
            </parent>
        </root>"""
        
        results = findall('<parent>/<child>', xml)
        self.assertEqual(len(results), 3)
        self.assertIn('First', results[0])
        self.assertIn('Second', results[1])
        self.assertIn('Third', results[2])

    def test_findall_empty(self):
        """Test findall with no matches"""
        xml = "<root><a>1</a><b>2</b></root>"
        
        results = findall('<nonexistent>', xml)
        self.assertEqual(results, [])
        
        results = findall('<root>/<nonexistent>', xml)
        self.assertEqual(results, [])

    def test_findall_malformed(self):
        """Test findall with malformed XML"""
        bad_xml = [
            "<root><a>1</a><b>2</b>",  # Missing end tag
            "<root><a>1</b></root>",    # Mismatched tags
            None,                        # None input
            "",                          # Empty input
        ]
        
        for xml in bad_xml:
            results = findall('<any>', xml)
            self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main() 