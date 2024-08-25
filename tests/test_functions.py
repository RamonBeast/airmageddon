import unittest
from unittest.mock import patch, MagicMock
from utils.functions import LLMFunctions

class TestLLMFunctions(unittest.TestCase):
    def setUp(self):
        self.llm_functions = LLMFunctions()

    def test_parse_response(self):
        response = '<function=next></function>'
        self.assertEqual(LLMFunctions._parse_response(response), ('next', {}))

        response = '<function=owners_away>{"away": true}</function>'
        self.assertEqual(LLMFunctions._parse_response(response), ('owners_away', {'away': True}))

        response = 'No function call'
        self.assertEqual(LLMFunctions._parse_response(response), (None, None))

    def test_call_function(self):
        with patch.object(self.llm_functions, 'next', return_value='Next function called') as mock_method:
            self.assertEqual(self.llm_functions.call_function('next', {}), 'Next function called')
            mock_method.assert_called_once()

        with patch.object(self.llm_functions, 'owners_away', return_value=None) as mock_method:
            self.assertEqual(self.llm_functions.call_function('owners_away', {'away': True}), None)
            mock_method.assert_called_once_with(away=True)

    def test_call_nonexistent_function(self):
        result = self.llm_functions.call_function('nonexistent_function', {})
        self.assertIsNone(result)

    def test_call_non_callable_attribute(self):
        result = self.llm_functions.call_function('value', {})
        self.assertIsNone(result)

    def test_is_function_call(self):
        response = '<function=next></function>'
        self.assertEqual(LLMFunctions.is_function_call(response), 'next')

        response = 'No function call'
        self.assertIsNone(LLMFunctions.is_function_call(response))

    def test_call_class_function(self):
        mock_instance = MagicMock()
        mock_instance.call_function.return_value = 'Function called'

        response = '<function=next></function>'
        self.assertEqual(LLMFunctions.call_class_function(mock_instance, response), 'Function called')
        mock_instance.call_function.assert_called_once_with('next', {})

    def test_next(self):
        with patch('utils.logger.Logger.info') as mock_logger:
            self.llm_functions.next()
            mock_logger.assert_called_once_with('LLM invoked next')

    def test_owners_away(self):
        with patch('utils.logger.Logger.info') as mock_logger:
            with patch.object(self.llm_functions._memory, 'create_memory') as mock_memory:
                self.llm_functions.owners_away(True)
                mock_logger.assert_called_once_with('LLM invoked owners_away = True')
                mock_memory.assert_called_once_with('OwnersAway', True)

if __name__ == '__main__':
    unittest.main()
