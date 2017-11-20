import xml.etree.ElementTree as ET

def parse_xml_to_dot(string_xml):
    xmlDocument = ET.fromstring(string_xml)
    dotString = 'digraph G {\n';

    for definitionElement in xmlDocument.findall("NETWORK/DEFINITION"):
        firstValue = definitionElement.find('GIVEN')
        secondValue = definitionElement.find('FOR')
        if firstValue != None and secondValue != None:
            dotString += '  "{}" -> "{}";\n'.format(firstValue.text, secondValue.text)

    dotString +='}\n'
    return dotString
