try:
    import configparser as cp
    from logging import config
    from os.path import exists
    import sys
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class ConfigHandler:
    new_config = False
    config_created = False

    def __init__(self, config_name="config.ini", my_path = ".") -> None:
        self.config = cp.RawConfigParser()
        self.config_name = config_name
        self.my_path = my_path

    def read_config(self) -> bool:
        if exists(self.config_name):
            self.config.read(self.config_name)
            self.config_created = True
            return True
        return False

    def write_config(self) -> bool:
        if self.config_created:
            with open(self.config_name, 'w') as config_file:
                self.config.write(config_file)
            return True
        return False

    def set_option(self, section_name : str, option_name : str, value) -> None:
        if type(value) is str:
            self.config.set(section_name, option_name, value)
        else:
            self.config.set(section_name, option_name, str(value))

    def get_option(self, section_name : str, option_name : str, value='no val') -> str:
        return self.config.get(section_name, option_name, fallback=value)

    def get_config_obj(self):
        return self.config

    def create_config(self, section_names : list) -> None:
        for name in section_names:
            self.config.add_section(name)

        self.config_created = True
        self.new_config = True    