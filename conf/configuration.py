import yaml
import os
from dotenv import load_dotenv
from utils.logger import Logger

class Configuration:
    def __init__(self, config_path: str = './conf/config.yml'):
        self.config_path = config_path
        self.config = {}
        self.config_loaded = False

        load_dotenv()
        self._load_conf()

    def _load_conf(self) -> bool:
        try:
            with open(self.config_path, 'r') as cfg:
                self.config = yaml.safe_load(cfg)

                if 'config' not in self.config:
                    Logger.error('Cannot find \'config\' section configuration file')
                    return False
                
                if 'agents' not in self.config:
                    Logger.error('Cannot find \'agents\' section configuration file')
                    return False
                
                self.config_loaded = True
        except IOError as e:
            Logger.error(f'Cannot open configuration file: {e}')
            return self.config_loaded
        
        return self.config_loaded

    def get_config_param(self, param: str) -> str | None:
        """ Return a parameter from the config section """
        if not self.config_loaded:
            return None
        
        conf = self.config['config']

        """ Return a parameter from the config section only """
        return os.getenv(param.upper(), conf[param] if param in conf else None)
    
    def get_config_bool(self, param: str) -> bool | None:
        """ Returns a boolean parameter from the config section """
        if not self.config_loaded:
            return None
        
        conf = self.config['config']

        par = os.getenv(param.upper(), conf[param] if param in conf else None)

        if isinstance(par, bool):
            return par
        elif isinstance(par, str):
            if par.lower() == 'true' or par.lower() == 'yes':
                return True
            else:
                return False
        else:
            return None

    def get_agent_config(self, agent: str) -> str | None:
        if not self.config_loaded:
            return None
        
        conf = self.config['agents']

        return conf[agent] if agent in conf else None

    def get_config(self) -> dict:
        return self.config