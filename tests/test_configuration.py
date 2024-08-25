import unittest
from unittest.mock import patch, mock_open
from utils.configuration import Configuration

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.config_path = './conf/config-sample.yml'
        self.config_content = 'config:\n  key: value\nagents:\n  agent1: details'
        self.invalid_config_content = 'invalid: yaml'
        self.conf_params = {
                'config': {
                    'param1': 'value1',
                    'param2': True,
                },
                'agents': {
                    'agent1': 'value1',
                },
                'debug': {
                    'active': True,
                    'param1': 'value1',
                    'param2': False,
                }
            }
        self.cfg = None
        Configuration._instance = None

    @patch('utils.configuration.yaml.safe_load')
    @patch('utils.configuration.load_dotenv')
    def test_init(self, mock_load_dotenv, mock_safe_load):
        mock_safe_load.return_value = {'config': {}, 'agents': {}}
        config = Configuration(self.config_path)
        mock_load_dotenv.assert_called_once()
        mock_safe_load.assert_called_once()
        self.assertTrue(config.config_loaded)

    @patch('utils.configuration.yaml.safe_load')
    @patch('utils.configuration.load_dotenv')
    def test_load_conf_success(self, mock_load_dotenv, mock_safe_load):
        mock_safe_load.return_value = {'config': {}, 'agents': {}}
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            config = Configuration(self.config_path)
            self.assertTrue(config.config_loaded)
            mock_file.assert_called_once_with(self.config_path, 'r')

    @patch('utils.configuration.yaml.safe_load')
    @patch('utils.configuration.load_dotenv')
    def test_load_conf_file_not_found(self, mock_load_dotenv, mock_safe_load):
        mock_safe_load.side_effect = IOError('File not found')
        config = Configuration(self.config_path)
        self.assertFalse(config.config_loaded)

    @patch('utils.configuration.yaml.safe_load')
    @patch('utils.configuration.load_dotenv')
    def test_load_conf_invalid_yaml(self, mock_load_dotenv, mock_safe_load):
        mock_safe_load.return_value = {}
        with patch('builtins.open', mock_open(read_data=self.invalid_config_content)) as mock_file:
            config = Configuration(self.config_path)
            self.assertFalse(config.config_loaded)

    @patch('utils.configuration.yaml.safe_load')
    @patch('utils.configuration.load_dotenv')
    def test_load_conf_missing_config_section(self, mock_load_dotenv, mock_safe_load):
        mock_safe_load.return_value = {'agents': {}}
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            config = Configuration(self.config_path)
            self.assertFalse(config.config_loaded)

    @patch('utils.configuration.yaml.safe_load')
    @patch('utils.configuration.load_dotenv')
    def test_load_conf_missing_agents_section(self, mock_load_dotenv, mock_safe_load):
        mock_safe_load.return_value = {'config': {}}
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            config = Configuration(self.config_path)
            self.assertFalse(config.config_loaded)

    def test_singleton(self):
        config1 = Configuration(self.config_path)
        config2 = Configuration(self.config_path)
        self.assertIs(config1, config2)

    @patch('utils.configuration.yaml.safe_load')
    def test_get_config_param(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.assertEqual(self.cfg.get_config_param('param1'), 'value1')
        self.assertIsNone(self.cfg.get_config_param('param3'))

    @patch('utils.configuration.yaml.safe_load')
    def test_get_config_bool(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params

        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.assertTrue(self.cfg.get_config_bool('param2'))
        self.assertFalse(self.cfg.get_config_bool('param1'))
        self.assertIsNone(self.cfg.get_config_bool('param3'))

    @patch('utils.configuration.yaml.safe_load')
    def test_get_agent_config(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.assertEqual(self.cfg.get_agent_config('agent1'), 'value1')
        self.assertIsNone(self.cfg.get_agent_config('agent2'))

    @patch('utils.configuration.yaml.safe_load')
    def test_get_section_config(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.assertEqual(self.cfg.get_section_config('debug', 'param1'), 'value1')
        self.assertIsNone(self.cfg.get_section_config('debug', 'param3'))
        self.assertIsNone(self.cfg.get_section_config('section2', 'param1'))

    @patch('utils.configuration.yaml.safe_load')
    def test_get_debug_param(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.assertEqual(self.cfg.get_debug_param('param1'), 'value1')
        self.assertIsNone(self.cfg.get_debug_param('param3'))

    @patch('utils.configuration.yaml.safe_load')
    def test_get_debug_bool(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.assertFalse(self.cfg.get_debug_bool('param2'))
        self.assertIsNone(self.cfg.get_debug_bool('param3'))

    @patch('utils.configuration.yaml.safe_load')
    def test_get_config(self, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)
            
        self.assertEqual(self.cfg.get_config(), self.conf_params)

    @patch('utils.configuration.yaml.safe_load')
    @patch('os.getenv')
    def test_get_config_param_env(self, mock_getenv, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        mock_getenv.return_value = 'env_value'
        self.assertEqual(self.cfg.get_config_param('param1'), 'env_value')

    @patch('utils.configuration.yaml.safe_load')
    @patch('os.getenv')
    def test_get_config_bool_env(self, mock_getenv, mock_safe_load):
        mock_safe_load.return_value = self.conf_params
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        mock_getenv.return_value = 'True'
        self.assertTrue(self.cfg.get_config_bool('param1'))

    @patch('utils.configuration.yaml.safe_load')
    def test_not_loaded(self, mock_safe_load):
        mock_safe_load.return_value = {}
        
        with patch('builtins.open', mock_open(read_data=self.config_content)) as mock_file:
            self.cfg = Configuration(self.config_path)

        self.cfg.config_loaded = False
        self.assertIsNone(self.cfg.get_config_param('param1'))
        self.assertIsNone(self.cfg.get_config_bool('param2'))
        self.assertIsNone(self.cfg.get_agent_config('agent1'))
        self.assertIsNone(self.cfg.get_section_config('debug', 'param1'))
        self.assertIsNone(self.cfg.get_debug_param('param1'))
        self.assertIsNone(self.cfg.get_debug_bool('param2'))


if __name__ == '__main__':
    unittest.main()
