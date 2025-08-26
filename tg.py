import requests

TOKEN = "8418144469:AAG87NmoRc-4Q1C04aTwzvYf6Xgv9GDuzt4"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(url)
updates = response.json()

for update in updates["result"]:
    if "message" in update:
        print("Ваш chat_id:", update["message"]["chat"]["id"])