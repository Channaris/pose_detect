import pymongo
import configparser

config = configparser.ConfigParser()
config.read('D:\\work\\braiven\\punching\\mediapipe\\pose_detect\\config.ini')

mongo = config['config']['mongo']
# print(mongo)

class savedata:
    def __init__(self):
        myclient = pymongo.MongoClient("mongodb+srv://wiewworkmotion:wiewworkmotion@cluster0.pu7bm.mongodb.net/?retryWrites=true&w=majority")
        mydb = myclient["pose_detect"]
        self.mycol = mydb["pose_detect"]

    def insert_data_mongo(self, data):

        mydict = data

        x = self.mycol.insert_one(mydict)