from flask import Flask
import numpy as np

app = Flask(__name__)


@app.before_first_request
def load_huge_file():
    loaded_data = np.load('../data/saved/doc_embeds_reshaped_1ep_0.5.npy')
    print("Huge data set loaded!")
    global vectors
    vectors = loaded_data


load_huge_file()


@app.route("/")
def calc():
    global vectors
    return 'using cached vectors, len=%s' % len(vectors)


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
