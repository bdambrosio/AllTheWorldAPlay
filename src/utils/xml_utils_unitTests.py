import unittest
from xml_utils import format_xml, XMLFormatError

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

if __name__ == '__main__':
    unittest.main() 