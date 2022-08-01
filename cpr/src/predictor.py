
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from google.cloud import storage
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor

class CprPredictor(SklearnPredictor):
    
    def __init__(self):
        return
    
    def load(self, artifacts_uri: str):
        """Loads the preprocessor artifacts."""
        super().load(artifacts_uri)
        gcs_client = storage.Client()
        with open("model_xgb.json", 'wb') as preprocessor_f:
            gcs_client.download_blob_to_file(
                f"{artifacts_uri}/model_xgb.json", preprocessor_f
            )

        with open("model_xgb.json", "rb") as f:
            bst = xgb.Booster(model_file=f)

        self._bst = bst
    
    def predict(self, instances):
        """Performs prediction."""
        instances = instances["instances"]
        data_dic = {feature:[v] for feature in columns for value in dict["instances"] for v in value if feature != "Cover_Type"}
        df = pd.DataFrame(data_dic)
        dtrain = xgb.DMatrix(df)
        outputs = self._bst.predict(dtrain)

        return {"predictions": outputs}
