from bs4 import BeautifulSoup as bs
from typing import Dict, List, Tuple


def bs_from_xml(annotation_path: str) -> bs:
    assert annotation_path.endswith(".xml"), 'read_label() -> annotation_path not xml file'

    with open(annotation_path, 'r') as f:
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)

    return bs(content)


def read_objects_label(annotation_path: str) -> List[Dict]:
    bs_content = bs_from_xml(annotation_path)
    objects = objects_from_bs(bs_content)
    return objects


def objects_from_bs(bs_content: bs) -> List[Dict]:
    objects_xml = bs_content.annotation.findAll('object')

    objects = []

    for obj in objects_xml:

        xmin = int(obj.xmin.contents[0])
        ymin = int(obj.ymin.contents[0])
        xmax = int(obj.xmax.contents[0])
        ymax = int(obj.ymax.contents[0])

        cls = obj.find('name').contents[0]

        objects.append({
            'cls': cls,
            'bbox': [xmin, ymin, xmax, ymax],
        })

    return objects

def size_from_bs(bs_content: bs) -> Tuple[int, int]:
    size_xml = bs_content.annotation.find('size')
    width = int(size_xml.width.contents[0])
    height = int(size_xml.height.contents[0])
    return width, height


def separate_objects(objects: List[Dict]) -> Tuple[List, List]:
    cls = []
    bbox = []

    for obj in objects:
        cls.append(obj['cls'])
        bbox.append(obj['bbox'])

    return cls, bbox


def annotation_stats(annotation_path: str) -> Tuple[List[Dict], Tuple[int, int]]:
    bs_content = bs_from_xml(annotation_path)
    objects = objects_from_bs(bs_content)
    size = size_from_bs(bs_content)
    return objects, size