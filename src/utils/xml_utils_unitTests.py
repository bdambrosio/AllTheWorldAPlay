import unittest
from xml_utils import format_xml, XMLFormatError, find, findall

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

    def test_findall_empty(self):
        """Test findall with no matches"""
        xml = "<root><a>1</a><b>2</b></root>"
        
        results = findall('<nonexistent>', xml)
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

    def test_findall_siblings(self):
        """Test findall with multiple siblings at same level"""
        xml = """<root>
            <parent>
                <child>First</child>
                <child>Second</child>
                <child>Third</child>
            </parent>
        </root>"""
        
        results = findall('<child>', xml)
        self.assertEqual(len(results), 3)
        self.assertIn('First', results[0])
        self.assertIn('Second', results[1])
        self.assertIn('Third', results[2])

    def test_findall_self_closing(self):
        """Test findall with self-closing tags"""
        xml = '<root><item/><item x="1"/><item>text</item></root>'
        results = findall('<item>', xml)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'text')

    def test_format_xml_self_closing(self):
        """Test formatting XML with self-closing tags"""
        xml = '<root><item/><data x="1" y="2"/></root>'
        expected = """<root>
  <item/>
  <data x="1" y="2"/>
</root>"""
        self.assertEqual(format_xml(xml), expected)

    def test_format_xml_mixed_tags(self):
        """Test formatting XML with mix of normal and self-closing tags"""
        xml = '<root><item/><data>content</data><empty/></root>'
        expected = """<root>
  <item/>
  <data>content</data>
  <empty/>
</root>"""
        self.assertEqual(format_xml(xml), expected)

    def test_find_self_closing(self):
        """Test finding self-closing tags"""
        xml = '<root><pos x="1" y="2"/><data>content</data></root>'
        self.assertEqual(find('<pos>', xml), '')  # Self-closing tag returns empty string
        self.assertEqual(find('<data>', xml), 'content')  # Normal tag returns content

    def test_find_multiple_self_closing(self):
        """Test finding self-closing tags with multiple instances"""
        xml = '<root><item/><item x="1"/><item>text</item></root>'
        self.assertEqual(find('<item>', xml), '')  # First match is self-closing
        
    def test_find_attributes_self_closing(self):
        """Test finding self-closing tags with attributes"""
        xml = '<root><pos x="1" y="2" type="point"/></root>'
        self.assertEqual(find('<pos>', xml), '')  # Self-closing with attributes returns empty string

if __name__ == '__main__':
    unittest.main() 