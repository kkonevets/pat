
import gensim
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient("fips")
db = client.fips

def