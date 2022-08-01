import tornado.web

class HealthCheckHandler(tornado.web.RequestHandler):
  # Health checks only need to respond to GET requests
  def get(self):
    ready = True # a place holder for the logic that
                 #   determines the health of the system
    if ready:
      # If everything is ready, respond with....
      self.set_status(200, 'OK')
      self.write('200: OK')
      self.write(json.dumps({"is_healthy": True}))
    else:
      # If everything is NOT ready, respond with....
      self.set_status(503, 'Service Unavailable')
      self.write('503: Service Unavailable')
      self.write(json.dumps({"is_healthy": False}))
    # finish the response
    self.finish()
    
class PredictionHandler(tornado.web.RequestHandler):
  def __init__(
    self,
    application: "Application",
    request: tornado.httputil.HTTPServerRequest,
  ) -> None:
    # We decided to load our model in the prediction
    #   constructor. This is not mandatory
    super().__init__(application, request, **kwargs)
 
    # deserialize a model that can serve predictions
    #   (we used .joblib, this is not mandatory)
    # NOTE: the model used in this demo is a simple
    #   decision tree that predicts on the iris dataset. 
    #   it's input expects a 2D array in the format 
    #   [[1,1,1,1], [2,2,2,2], [3,3,3,3], ...]
    import xgboost as xgb
    
    self.model = xgb.Booster(model_file='./model.json') 
 
  # PredictionHandler only responds with predictions when 
  #   called from a POST request
  def post(self):
    response_body = None
    try:
      # get the request body
      req_body = tornado.escape.json_decode(self.request.body)
       
      # get the instances from the body
      instances = req_body.get("instances", {})
       
      # if parameters don't exist, create a dummy dictionary
      parameters = req_body.get("parameters", {})
     
      # generate our predictions for each instance
      # NOTE: .tolist() is specific to our implementation 
      #   as it matches the model we built
      predictions = self.model.predict(instances).tolist()
       
      # there may need to be extra steps to make sure the
      #   response is properly formatted (ex. .tolist() above)
      response_body = json.dumps({'predictions': predictions})
  
    # catch errors
    except Exception as e:
      response_body = json.dumps({'error:':str(e)})
 
    # set up the response
    self.set_header("Content-Type", "application/json")
    self.set_header("Content-Length", len(response_body))
    self.write(response_body)
       
    # send the response
    self.finish()

def make_app():
  # Create the app and assign the handlers to routes
  tornado_app = tornado.web.Application([('/health_check', HealthCheckHandler),
                 ('/predict', PredictionHandler)],
               debug=False)
  tornado_app.listen(8080)
  tornado.ioloop.IOLoop.current().start()
   
 
if __name__ == '__main__':
  make_app()
