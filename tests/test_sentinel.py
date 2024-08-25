import unittest
from unittest.mock import patch, Mock
from sentinel import Sentinel
from utils.logger import Logger
from utils.functions import LLMFunctions

class TestSentinel(unittest.TestCase):
    def setUp(self):
        self.sentinel = Sentinel(max_memories=5)

    def tearDown(self):
        del self.sentinel

    @patch('agents.guard.Guard.send_message')
    @patch('agents.burglar.Burglar.send_message')
    @patch('utils.functions.LLMFunctions.is_function_call')
    def test_analyze_feed_normal_case(self, mock_is_function_call, mock_burglar_send_message, mock_guard_send_message):
        mock_guard_send_message.side_effect = ['Guard response', 'Final decision']
        mock_burglar_send_message.return_value = 'Ex-burglar response'
        mock_is_function_call.side_effect = [False, True]

        result = self.sentinel.analyze_feed('Normal camera feed')

        self.assertEqual(result, 'Final decision')
        mock_guard_send_message.assert_called_with('Guard: Guard response\nCameraFeed: Normal camera feed\nEx-burglar: Ex-burglar response\n')
        mock_burglar_send_message.assert_called_with('Guard: Guard response\nCameraFeed: Normal camera feed\n')
        mock_is_function_call.assert_called_with('Final decision')

    @patch('agents.guard.Guard.send_message')
    @patch('utils.functions.LLMFunctions.is_function_call')
    def test_analyze_feed_error_case(self, mock_is_function_call, mock_guard_send_message):
        mock_guard_send_message.return_value = None
        mock_is_function_call.return_value = False

        with patch.object(Logger, 'error') as mock_error:
            result = self.sentinel.analyze_feed('Error camera feed')

        self.assertIsNone(result)
        mock_guard_send_message.assert_called_once_with('CameraFeed: Error camera feed')
        mock_error.assert_called_once_with('Error sending message')

    @patch('agents.burglar.Burglar.get_cumulative_tokens')
    @patch('agents.guard.Guard.get_cumulative_tokens')
    def test_get_cumulative_tokens(self, mock_guard_tokens, mock_burglar_tokens):
        mock_guard_tokens.return_value = {'completion_tokens': 10, 'prompt_tokens': 20}
        mock_burglar_tokens.return_value = {'completion_tokens': 5, 'prompt_tokens': 15}

        result = self.sentinel.get_cumulative_tokens()

        self.assertEqual(result, {'completion_tokens': 15, 'prompt_tokens': 35})

if __name__ == '__main__':
    unittest.main()
