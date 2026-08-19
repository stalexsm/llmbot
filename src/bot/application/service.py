"""The ``ProcessUserMessage`` use case."""

import time

import structlog

from bot.application.errors import ApplicationError, EmptyInferenceResponseError
from bot.application.models import UserMessageRequest, UserMessageResponse
from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceRequest
from bot.inference.provider import InferenceProvider


class ApplicationService:
    """Processes a single user message.

    The service is stateless: every message is an independent inference
    request, no conversation history is stored or reused.
    """

    def __init__(
        self,
        inference: InferenceProvider,
        model: ModelId,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._inference = inference
        self._model = model
        self._logger = logger.bind(component="application_service")

    async def process_message(self, request: UserMessageRequest) -> UserMessageResponse:
        inference_request = self._build_inference_request(request)
        started_at = time.monotonic()
        try:
            inference_response = await self._inference.generate(inference_request)
        except ApplicationError as exc:
            self._log_inference(
                request.request_id,
                started_at,
                status="error",
                error=type(exc).__name__,
            )
            raise
        if not inference_response.content.strip():
            self._log_inference(request.request_id, started_at, status="empty_response")
            raise EmptyInferenceResponseError("Inference provider returned an empty response")
        self._log_inference(request.request_id, started_at, status="success")
        return UserMessageResponse(
            request_id=inference_response.request_id,
            text=inference_response.content,
        )

    def _build_inference_request(self, request: UserMessageRequest) -> InferenceRequest:
        return InferenceRequest(
            request_id=request.request_id,
            model=self._model,
            messages=(InferenceMessage(role=MessageRole.USER, content=request.text),),
        )

    def _log_inference(
        self,
        request_id: RequestId,
        started_at: float,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        event: dict[str, str | int] = {
            "request_id": request_id,
            "model": self._model,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "status": status,
        }
        if error is not None:
            event["error"] = error
        self._logger.info("inference_finished", **event)
