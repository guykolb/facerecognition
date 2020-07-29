import urllib.request

def DownloadPhoto(photo_URL, photo_name):
    urllib.request.urlretrieve(photo_URL, photo_name + ".jpg")
