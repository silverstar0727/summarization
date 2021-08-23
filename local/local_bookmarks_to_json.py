import chrome_bookmarks
import json

data = chrome_bookmarks.folders

with open("bookmarks.json", "w") as f:
    json.dump(data, f)