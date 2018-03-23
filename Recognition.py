from flask import Flask
from flask import request
import cv2
from ctcfunc import getCode
import json
import numpy as np

app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def home():
    return '<h1>Home</h1>'


@app.route('/breakcaptcha', methods = ['POST'])
def captcha_reg():
    # post the image
    res = {}
    res['type'] = False
    res['result'] = ""

    m = request.form['img']
    if m is not None:
        print(m)
        c = getCode(m)
        if c is not None:
            res['result'] = c
            res['type'] = True
    # jsonstr = ""
    jsonstr=json.dumps(res)
    return jsonstr


@app.route('/signin', methods = ['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''


@app.route('/signin', methods = ['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username'] is not '':
        return getCode(request.form['username'])
    else:
        return 'Error'
        # if request.form['username']=='admin' and request.form['password']=='password':
        #     return '<h3>Hello, admin!</h3>'
        # return '<h3>Bad username or password.</h3>'


if __name__ == '__main__':
    app.run()
