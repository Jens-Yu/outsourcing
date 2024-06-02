import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi

# MongoDB URI and Database Name
MONGO_URI = 'mongodb+srv://yanghan:yh123456@yh.psdsh48.mongodb.net/?retryWrites=true&w=majority&appName=yh'
DATABASE_NAME = 'yh'

async def clear_collections():
    client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi('1'))
    db = client[DATABASE_NAME]

    # List of collections you want to clear
    collections_to_clear = ['productions']

    try:
        # Ping the database to check if connection is successful
        await db.command('ping')
        print("Connection to MongoDB successful!")

        # Loop through the list and delete all documents from each collection
        for collection_name in collections_to_clear:
            result = await db[collection_name].delete_many({})
            print(f"Documents deleted from {collection_name}: {result.deleted_count}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the connection to MongoDB
        client.close()

# Run the async function using asyncio
if __name__ == '__main__':
    asyncio.run(clear_collections())
