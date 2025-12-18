"""Mock for HuggingFace Inference API client."""


class MockInferenceClient:
    """
    Mock HF InferenceClient for testing API interactions.

    Usage:
        # Success case
        client = MockInferenceClient(response=[{'label': 'toxic', 'score': 0.95}])

        # Failure case
        client = MockInferenceClient(error=Exception("Rate limit"))

        # Multiple calls (retry scenarios)
        client = MockInferenceClient(responses=[
            Exception("Fail 1"),
            Exception("Fail 2"),
            [{'label': 'toxic', 'score': 0.8}]  # Third call succeeds
        ])
    """

    def __init__(self, response=None, error=None, responses=None):
        """
        Initialize mock client.

        Args:
            response: Single successful response
            error: Single error to raise
            responses: List of responses/errors for multiple calls
        """
        self.call_count = 0

        if responses is not None:
            # Multiple responses for retry scenarios
            self.responses = responses
        elif error is not None:
            self.responses = [error]
        elif response is not None:
            self.responses = [response]
        else:
            # Default: non-toxic response
            self.responses = [[{"label": "non-toxic", "score": 0.05}]]

    def text_classification(self, text, model=None):
        """
        Mock text classification method.

        Args:
            text: Text to classify
            model: Model to use (ignored in mock)

        Returns:
            Classification result or raises exception

        Raises:
            Exception: If current response is an exception
        """
        idx = min(self.call_count, len(self.responses) - 1)
        result = self.responses[idx]
        self.call_count += 1

        if isinstance(result, Exception):
            raise result
        return result
