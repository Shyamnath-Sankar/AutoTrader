import http.client

conn = http.client.HTTPSConnection("ultimate-economic-calendar.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "9f51bde570msh94e7405e3aea572p1292fejsn66f7a517d5b4",
    'x-rapidapi-host': "ultimate-economic-calendar.p.rapidapi.com",
    'Content-Type': "application/json"
}

conn.request("GET", "/economic-events/tradingview?from=2024-06-17&to=2024-06-19&countries=US%2C%20DE", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))