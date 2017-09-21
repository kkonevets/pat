import socket
import threading
import SocketServer
import gc

import numpy as np
import pandas as pd
from prediction import predict
import ast


from messaging import send_msg, recv_msg

VEC_PATH = '/data/saved/doc_embeds_reshaped_1ep_0.5.npy'
IDS_PATH = '/data/saved/fnames_1ep_0.5.npy'


class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        keys = recv_msg(self.request)
        keys = ast.literal_eval(keys)

        preds = predict(vectors, vectors_norm_squared, names, keys, lim=200)

        response = str(preds)
        send_msg(self.request, str(preds))

        cur_thread = threading.current_thread()
        print(cur_thread.name)
        gc.collect()

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "localhost", 5000

    vectors = np.load(VEC_PATH)
    # precompute dot products
    vectors_norm_squared = np.square(vectors).sum(axis=1)

    ids = np.load(IDS_PATH)
    names = pd.Series(range(len(ids)), index = ids)


    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    print('server started')
    server.serve_forever()
