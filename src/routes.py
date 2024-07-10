from views import index, js_handler, training_handler, css_handler, classifier_handler

def setup_routes(app):
	app.router.add_get('/', index)
	app.router.add_get('/script.js', js_handler)
	app.router.add_get('/style.css', css_handler)
	app.router.add_post('/train', training_handler)
	app.router.add_post('/predict', classifier_handler)
