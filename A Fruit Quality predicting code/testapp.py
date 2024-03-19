from flask import Flask

app=Flask(__name__)

@app.route('/fruit')
def test():
    return "hello world"
