
import logging
import os
import traceback

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
import uvicorn

from predictor import CprPredictor
from google.cloud.aiplatform import prediction

class ModelServer:
    """Model server to do custom prediction routines."""

    def __init__(self):
        """Initializes a fastapi application and sets the configs.

        Args:
            handler (Handler):
                Required. The handler to handle requests.
        """
        self._init_logging()

        self.handler = prediction.handler.PredictionHandler(
            os.environ.get("AIP_STORAGE_URI"), predictor=CprPredictor,
        )

        if "AIP_HTTP_PORT" not in os.environ:
            raise ValueError(
                "The environment variable AIP_HTTP_PORT needs to be specified."
            )
        if (
            "AIP_HEALTH_ROUTE" not in os.environ
            or "AIP_PREDICT_ROUTE" not in os.environ
        ):
            raise ValueError(
                "Both of the environment variables AIP_HEALTH_ROUTE and "
                "AIP_PREDICT_ROUTE need to be specified."
            )
        self.http_port = int(os.environ.get("AIP_HTTP_PORT"))
        self.health_route = os.environ.get("AIP_HEALTH_ROUTE")
        self.predict_route = os.environ.get("AIP_PREDICT_ROUTE")

        self.app = FastAPI()
        self.app.add_api_route(
            path=self.health_route, endpoint=self.health, methods=["GET"],
        )
        self.app.add_api_route(
            path=self.predict_route, endpoint=self.predict, methods=["POST"],
        )

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)

    def _init_logging(self):
        """Initializes the logging config."""
        logging.basicConfig(
            format="%(asctime)s: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
            level=logging.INFO,
        )

    def health(self):
        """Executes a health check."""
        return {}

    async def predict(self, request: Request) -> Response:
        """Executes a prediction.

        Args:
            request (Request):
                Required. The prediction request.

        Returns:
            The response containing prediction results.
        """
        try:
            return await self.handler.handle(request)
        except HTTPException:
            # Raises exception if it's a HTTPException.
            raise
        except Exception as exception:
            error_message = "An exception {} occurred. Arguments: {}.".format(
                type(exception).__name__, exception.args
            )
            logging.info(
                "{}\nTraceback: {}".format(error_message, traceback.format_exc())
            )

            # Converts all other exceptions to HTTPException.
            raise HTTPException(status_code=500, detail=error_message)
