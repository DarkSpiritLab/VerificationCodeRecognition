import requests
import base64
url="http://www.cspro.org/lead/servlet/ImageServlet"
a=requests.get(url)
s=a.content
# print(type(bytes(s)))
# print(bytes((s)))
url="http://127.0.0.1:5000/breakcaptcha"
p={}
print(type(base64.b64encode(s)))
p['img']=str(base64.b64encode(s),'utf-8')

t=requests.post(url,p)
print(t.content)
