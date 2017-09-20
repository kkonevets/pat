from messaging import send_msg, recv_msg
import socket
import ast

IP, PORT = 'localhost', 5000


def send(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        send_msg(sock, message)
        response = recv_msg(sock)
    finally:
        sock.close()
    return response

if __name__ == "__main__":
    keys = ['5984c4f5b6b1134b90638519', '5984c0e8b6b1131d3d638515', 
        '5984c6dcb6b11360626384fc', '5984d1feb6b11349ac638526', 
        '5984b5a7b6b113058b638539', '5984c20cb6b1132c976384f0']

    message = str(keys[:2])
    for i in range(20):
        response = send(IP, PORT, message)
        preds = ast.literal_eval(response)
        print("Received: %s" % preds.keys())

