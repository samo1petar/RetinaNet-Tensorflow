import os
from lib.tools.annotation import bs_from_xml, objects_from_bs


labels_dir = '/home/david/Projects/Retinus/dataset/annotations_all'

for xml_file in os.listdir(labels_dir):
    for obj in objects_from_bs(bs_from_xml(os.path.join(labels_dir, xml_file))):
        if obj['cls'] == 'w':
            print (xml_file)
            # os.remove(os.path.join(labels_dir, xml_file))