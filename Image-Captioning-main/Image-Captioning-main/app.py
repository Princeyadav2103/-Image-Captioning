from flask import Flask, render_template, redirect, request

import Caption_it

#__name == __main__
app = Flask(__name__) #instance of Flask passing module name

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/',methods = ['POST'])
def home():
	if request.method == 'POST':
	
		f = request.files['userfile']
		#print(f)
		path = "./static/{}".format(f.filename)
		f.save(path)
		caption = Caption_it.caption_this_image(path)
		#print(caption)

		result_dic = {
		'image' : path,
		'caption' : caption
		}

	return render_template("index.html", your_result = result_dic)


if __name__ == '__main__':
	app.run(debug=True) #debug=False, threaded=False