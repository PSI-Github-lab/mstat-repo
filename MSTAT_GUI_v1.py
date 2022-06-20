try:
    import sys
    from datetime import datetime
    import os
    from mstat.dependencies.helper_funcs import *
    from PyQt5 import QtCore, QtWidgets
    from mstat.gui.mstat_ctrl import MStatCtrl
    from mstat.dependencies.config_handler import ConfigHandler
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

if __name__ == "__main__":
    # check for known paths
    path = '.'
    config_hdlr = ConfigHandler(config_name='mstat_config.ini')
    if not config_hdlr.read_config():
        config_hdlr.create_config(['MAIN'])
        config_hdlr.set_option('MAIN', 'last start time', str(datetime.now()))
        config_hdlr.set_option('MAIN', 'train directory', path)
        config_hdlr.set_option('MAIN', 'test directory', path)
        config_hdlr.set_option('MAIN', 'windows num bits', '64' if sys.maxsize > 2**32 else '32')
        config_hdlr.set_option('MAIN', 'msfilereader installed', str(False))
        config_hdlr.set_option('MAIN', 'msconvert directory', '')

        config_hdlr.write_config()

    # open the gui
    app = QtWidgets.QApplication(sys.argv)
    mstat = MStatCtrl(app, root_path=path, config_hdlr=config_hdlr)

    sys.exit(app.exec_())