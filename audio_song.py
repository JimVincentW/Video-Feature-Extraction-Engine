import asyncio
from shazamio import Shazam

async def shazam(audio):
    loop = asyncio.get_event_loop()
    shazam_client = Shazam()
    
    out = await shazam_client.recognize_song(audio)
    if out["matches"] is None:
        return "No song found"
    else:
        return out
