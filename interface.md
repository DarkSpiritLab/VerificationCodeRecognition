## captcha recognition interface
- breakcaptcha
	 - request
		 - mode['POST']
		 - img=str(base64.b64encode(imgcontent),'utf-8')
	 - response
		 - {'type':'False'/'True'
				 'result':'abcd'
			}
-
