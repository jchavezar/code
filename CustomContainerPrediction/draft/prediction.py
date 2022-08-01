import tornado.web
import json
import pandas as pd
import xgboost as xgb

class PreprocessPrediction():
    def preproces(self, prediction_input):
        instances = prediction_input["instances"]
        df = pd.DataFrame(instances, columns=self._features)
        
        return xgb.DMatrix(df)
    
    def model_load(self, model: str):
        self.model = xgb.Booster(model_file='./model.json')
        return self.model
    
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
        
        self._features = ['Id','Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                          'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
                          'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
                          'Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19', 
                          'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
                          'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
        #Load our model in the prediction
        super().__init__(application, request)
    
 
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