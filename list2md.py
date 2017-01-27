import cgitb
import requests
import time
import config as cf

cgitb.enable(format='text')

result=[]

head = '# Top Deep Learning Projects\n' \
        'A list of popular github projects related to deep learning (ranked by stars automatically).\n'\
        'Please update list.txt (via pull requests)\n\n' \
        '| Project Name| Stars | Description | Updated/Created\n' \
        '| ------- | ------ | ------ | ------ \n'

tail = '\nLast Automatic Update: ' + time.strftime("%c") + \
       '\n\nInspired by https://github.com/aymericdamien/TopDeepLearning'

with open("list.txt") as f:
    list = f.readlines()
    for url in list:
        if url.startswith("https://github.com/"):
            api_url = "https://api.github.com/repos/" + url[19:].strip().strip("/")
            api_url = api_url + '?access_token=' + cf.ACCESS_TOKEN
            print(api_url)
            r = requests.get(url=api_url)
            r_json = r.json()
            if 'name' in r_json:
                result.append({'name': r_json['name'],
                               'description': r_json['description'],
                               'forks': r_json['forks_count'],
                               'created': r_json['created_at'],
                               'updated': r_json['updated_at'],
                               'url': r_json['html_url'],
                               'stars': r_json['stargazers_count']})


sorted_result = sorted(result, key=lambda repos: repos['stars'], reverse=True)
print(sorted_result)

def xstr(s):
    if s is None:
        return ''
    return str(s)

with open("README.md", 'w') as out:
    out.write(head)
    for repos in sorted_result:
        out.write('| [' +
                  repos['name'] + '](' + repos['url']  + ') | ' +
                  xstr(repos['stars']) + ' | ' + xstr(repos['description']) + ' | ' +
                  xstr(repos['updated']) + ' / ' + xstr(repos['created']) + ' \n')
    out.write(tail)
