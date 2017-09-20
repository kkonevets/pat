from flask import Flask
import numpy as np
from prediction import predict
from client

app = Flask(__name__)


@app.route("/")
def calc():
    global vectors, ids
    keys = ids[:1]

    response = client.send(client.IP, client.PORT, str(keys))
    # preds = ast.literal_eval(response)
    
    return response

    
if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
