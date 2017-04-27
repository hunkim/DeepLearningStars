"""
Multiprocess version of list2md.py

Examples
----------

    $ python list2md.multiprocess.py

"""
import requests
import time
import config
from multiprocessing.pool import Pool


def write_md(dict_list, filepath="README.md"):
    """Given a list of dict, write a markdown file

    Parameters
    ----------
    dict_list : list
        Each element is an dictionary of
        {"name": "Tensorflow",
         "url": "https://github.com/tensorflow/tensorflow",
         "stars": 55359,
         "description": "Computation using data flow graph ..."}

    filepath : str
        Readme path

    Returns
    ----------
    bool
        Returns True If everything went smooth
    """

    HEAD = """# Top Deep Learning Projects

A list of popular github projects related to deep learning (ranked by stars automatically).


Please update list.txt (via pull requests)


|Project Name| Stars | Description |
| ---------- |:-----:| ----------- |
"""

    TAIL = f"""

Last Automatic Update: {time.strftime("%c")}

Inspired by https://github.com/aymericdamien/TopDeepLearning
"""

    # sort descending by n_stars
    dict_list = sorted(dict_list, key=lambda x: x['stars'], reverse=True)

    # each data is a string (see `dict2md`)
    data_list = list(map(dict2md, dict_list))

    with open(filepath, 'w') as out:

        out.write(HEAD)
        out.write("\n".join(data_list))
        out.write(TAIL)

        return True


def dict2md(dict_):
    """Convert a dictionary to a markdown format"""
    return "| [{name}]({url}) | {stars} | {description} |".format(**dict_)


def get_url_list(filepath="list.txt"):
    """Read list.txt and returns a list of API urls"""

    def preprocess_url(url):
        """Returns an API url"""
        return "https://api.github.com/repos/" + url[19:].strip().strip("/")

    with open(filepath, 'r') as f:
        data = f.readlines()

    return list(map(preprocess_url, data))


def grab_data(url):
    """Go to the URL and grab a data

    Parameters
    ----------
    url : str
        URL to a github repo

    Returns
    ----------
    dict
        dict_keys(['name',
                   'description',
                   'forks',
                   'created',
                   'updated',
                   'url',
                   'stars'])
    """
    params = {
        "access_token": config.ACCESS_TOKEN
    }

    try:
        data_dict = requests.get(url, params=params).json()

        return {'name': data_dict['name'],
                'description': data_dict['description'],
                'forks': data_dict['forks_count'],
                'created': data_dict['created_at'],
                'updated': data_dict['updated_at'],
                'url': data_dict['html_url'],
                'stars': data_dict['stargazers_count']}

    except KeyError:
        raise Exception(f"{data_dict}")


def main():
    url_list = get_url_list()

    pool = Pool(processes=config.N_WORKERS)
    result = pool.map_async(grab_data, url_list)

    write_md(result.get())


if __name__ == '__main__':
    main()
