from flask import Flask
import client

app = Flask(__name__)


@app.route("/")
def calc():
    keys = ['5984c4f5b6b1134b90638519']

    response = client.send(client.IP, client.PORT, str(keys))
    # preds = ast.literal_eval(response)
    
    return response


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
