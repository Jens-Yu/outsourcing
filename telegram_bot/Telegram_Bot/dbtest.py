import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def main():
    uri = "mongodb+srv://yanghan:yh123456@yh.psdsh48.mongodb.net/?retryWrites=true&w=majority&appName=yh"
    client = AsyncIOMotorClient(uri)
    db = client.yh
    users_collection = db.users

    cursor = users_collection.find({'role': 'staff'})
    staff_members = await cursor.to_list(length=100)
    for member in staff_members:
        print(member)

asyncio.run(main())