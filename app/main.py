from flask import Flask
import numpy as np

app = Flask(__name__)

FNAME = '../data/saved/doc_embeds_reshaped_1ep_0.5.npy'

@app.before_first_request
def load_huge_file():
    global vectors
    vectors = np.load(FNAME, mmap_mode='r')
    print("Huge data set loaded!")


load_huge_file()


@app.route("/")
def calc():
    global vectors
    l = len(vectors)
    return 'using cached vectors, len=%s' % l


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
