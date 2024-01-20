#from: https://github.com/RiddlerQ/simple_image_download

from simple_image_download import Downloader
import datetime

starttime = datetime.datetime.now()

#class instantiation
response = Downloader()

searchwords = "rosemary-herb thyme-herb sage-herb" #can list out multiple keywords separated by space in between (within the string)
response.download(searchwords, limit=1000)

endtime = datetime.datetime.now()
print("Time elapsed:", (endtime - starttime))
print("Done")

