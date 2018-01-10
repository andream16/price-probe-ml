import json


def get_configuration(flag):

    data = json.load(open('configuration/conf.json'))
    if flag == 'remote':
        host = data['Remote']['Host']
        port = data['Remote']['DBPort']
        server_port = data['Remote']['ServerPort']
    else:
        host = data['Local']['Host']
        port = data['Local']['DBPort']
    db_url = data['Spark']['DBUrl']
    return host, port, db_url, server_port
