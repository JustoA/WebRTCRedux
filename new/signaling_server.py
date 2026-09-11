#!/usr/bin/env python

"""Echo server using the asyncio API."""

import asyncio
import json
from websockets.asyncio.server import serve
from itertools import count

# store connected clients as a dictionary where key = client ip, and values = tuple of (ip address, ports in use)

# Change this to whatever port range you want.
PORTS_START_AT = 40000

# Server prescribes what port to listen on

"""
You get the port number to listen on like so:
PORTS_START_AT + other_client_id
So clients 1 and 2 will listen on 40002 and 40001 respectively
By virtue of UDP, we dont really have to worry too much about connections starting at different times


Possible WS messages:
Name - Direction - Purpose - JSON
Hello - Client -> Server - Client announcing that it has joined - {type:'HELLO', ip: ip_address}
Welcome - Server -> Client - Server informs of clients and ports - {type: 'WELCOME', port: starting_port, clients: [(client,id)...]}
New Friend - Server -> Client - Server informs existing clients that a new member has joined - {type: 'NEWFRIEND', ip: ip_address, id:  id}
Bye - Client -> Server - Client leaves - {type: 'BYE', ip_address}
Client Left - Server -> Client - Server letting clients know someone disconnected - {'CLIENTLEFT', id}

"""
#                     ip_addr, id
connected_clients: dict[str, int] = {}
connected_clients_websockets = {}
id_gen = count(start=1)

async def send_welcome_message(websocket):
    msg = {"type":"WELCOME", "port":PORTS_START_AT, "clients":connected_clients}
    await websocket.send(json.dumps(msg))

async def send_client_left_message():
    pass

async def send_new_friend_message(client_ip_addr: str, new_client_id: int):
    msg = {"type":"NEWFRIEND","ip":client_ip_addr,"id":new_client_id}
    for client, ws in connected_clients_websockets.items():
        if client == client_ip_addr:
            continue # dont connect to yourself
        else:
            await ws.send(json.dumps(msg))


async def on_message(websocket):
    print("connected clients:", connected_clients)
    async for message in websocket:
        try:
            msg_json = json.loads(message)
            msg_type = msg_json["type"]
            if msg_type == 'HELLO':
                print("Got hello!")
                client_ip_addr = msg_json["ip"] 
                new_client_id = next(id_gen)
                await send_welcome_message(websocket)

                connected_clients[client_ip_addr] = new_client_id

                connected_clients_websockets[client_ip_addr] = websocket

                await send_new_friend_message(client_ip_addr, new_client_id)


            if msg_type == 'BYE':
                left_client_ip = msg_json["ip_address"]
                client_that_left = connected_clients.pop(left_client_ip)
                pass # todo

        except Exception as e:
            print("ERROR: ", str(e))
        await websocket.send(message)


async def main():
    server = await serve(on_message, "localhost", 8765)
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
