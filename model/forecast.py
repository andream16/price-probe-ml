from datetime import datetime
import json
import requests


class ForecastEntry:

    def __init__(self, date: str, price: float):
        self.date = date
        self.price = price


def save_forecast(item: str, forecast_dictionary, config):
    forecast_dict = {'name': forecast_dictionary['name'], 'item': item, 'forecast_entries': []}

    date_forecast = forecast_dictionary['date_forecast'].tolist()
    forecast_entries = forecast_dictionary['forecast'].tolist()

    for idx, val in enumerate(forecast_entries):
        date_str = datetime.fromtimestamp(date_forecast[idx] / 1000000000).strftime("%Y-%m-%d")
        current_price = float("{0:.2f}".format(val))
        forecast_dict['forecast_entries'].append({
            'date': date_str,
            'price': current_price,
            'score': forecast_dictionary['score'],
            'test_size': forecast_dictionary['percentage']
        })
    post_forecast(forecast_dict, config)


def post_forecast(forecast_dict, config):
    json_entry = json.dumps(forecast_dict, ensure_ascii=False)
    remote_url = 'http://' + config[0] + ':' + config[3] + '/api/statistics'
    r = requests.post(remote_url, data=json_entry)
    if r.status_code == 200:
        return
    else:
        print('Error while posting forecast.')
