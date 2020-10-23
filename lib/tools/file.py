import os


def mkdir(dir_path: str) -> None:
    if not os.path.exists(dir_path) and os.path.exists(dir_path.rsplit('/', 1)[0]):
        os.mkdir(dir_path)


def choose_one_from_dir(directory: str, pattern: str='') -> str:
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