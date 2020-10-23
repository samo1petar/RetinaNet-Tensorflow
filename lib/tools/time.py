import datetime


def get_time() -> str:
    time = datetime.datetime.now()
    return '{}-{}-{}_{}-{}-{}'.format(
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute,
        time.second,
    )