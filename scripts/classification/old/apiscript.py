import requests

url = "https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_timeseries_tas_timeseries_annual_1901-2024_mean_historical_cru_ts4.09_mean/USA?_format=json"
response = requests.get(url)
data = response.json()
print(data)