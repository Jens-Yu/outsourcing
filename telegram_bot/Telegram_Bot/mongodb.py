from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from setting import MONGO_URI

# Create a new client and connect to the server
client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client['yh']

admins_collection = db['admins']
users_collection = db['users']
quota_collection = db['quota']
productions_collection = db['productions']
operations_collection = db['operations']
