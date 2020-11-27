import datetime


def get_time() -> str:
    """
    Return strings like 2020-10-28_8-0-14
    """
    time = datetime.datetime.now()
    return '{}-{}-{}_{}-{}-{}'.format(
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute,
        time.second,
    )