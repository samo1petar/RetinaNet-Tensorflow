import json
import os
from random import shuffle
from typing import Any, Dict, List, Tuple, Union


def mkdir(dir_path: str) -> None:
    """
    Create new directory
    """
    if not os.path.exists(dir_path) and os.path.exists(dir_path.rsplit('/', 1)[0]):
        os.mkdir(dir_path)


def choose_one_from_dir(directory: str, pattern: str='') -> str:
    """
    Gives a user a choice to choose between subdirectories
    """
    assert os.path.exists(directory)
    files = [x for x in os.listdir(directory) if pattern in x]
    files.sort()
    for i, file in enumerate(files):
        print('[{}] - {}'.format(i + 1, file.rsplit('.', 1)[0]))
    while 1:
        index = input('Choose one from above: ')
        try:
            index = int(index)
        except:
            print('Numbers only please.')
            continue
        if index > 0 and index < len(files) + 1:
            return os.path.join(directory, files[index - 1])


def choose_model(directory: str) -> str:
    """
    Gives a user a choice to choose between models
    """
    assert os.path.exists(directory)
    models = [x.split('.index', 1)[0] for x in os.listdir(directory) if '.index' in x]
    models.sort()
    for i, file in enumerate(models):
        print('[{}] - {}'.format(i + 1, file.rsplit('.', 1)[0]))
    while 1:
        index = input('Choose one from above: ')
        try:
            index = int(index)
        except:
            print('Numbers only please.')
            continue
        if index > 0 and index < len(models) + 1:
            return os.path.join(directory, models[index - 1])


def shuffle_paths(paths1: List[str], paths2: List[str]) -> Tuple[List, List]:
    paths_combined = list(zip(paths1, paths2))
    shuffle(paths_combined)
    paths1, paths2 = zip(*paths_combined)
    return paths1, paths2


def get_annotation_and_image_paths(dataset_path: str, images_at: str, annotations_at: str) -> Tuple[List[str], List[str]]:

    images_dict = {}
    for root, _, files in os.walk(os.path.join(dataset_path, images_at)):
        for file in files:
            image_path_split = os.path.join(root, file).rsplit('.', 1)
            images_dict[image_path_split[0]] = image_path_split[1]

    annotations = []
    images = []
    for root, _, files in os.walk(os.path.join(dataset_path, annotations_at)):
        for file in files:
            annotation_full_path = os.path.join(root, file)
            annotations.append(annotation_full_path)
            annotation_path_split = annotation_full_path.rsplit('.', 1)

            image_path_without_extension = annotation_path_split[0].replace(annotations_at, images_at)
            image_path = image_path_without_extension + '.' + images_dict[image_path_without_extension]
            images.append(image_path)

    return annotations, images


def save_json(path: str, data1: Dict[str, Union[str, List[str]]]) -> None:
    with open(path, 'w') as f:
        json.dump(data1, f, indent=4)


def load_json(path: str) -> Any:
    with open(path) as f:
        data = json.load(f)
    return data
