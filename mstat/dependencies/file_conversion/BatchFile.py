from subprocess import Popen, PIPE

class BatchFile:
    ''' 
    This class runs batch files in a python environment. It handles passing arguments from python to the batch file call. 
    Dependencies: subprocess
    '''
    bat_name: str
    args = []
    cmd: str

    def __init__(self, bat_name) -> None:
        self.bat_name = bat_name
    
    def run(self, args, verbose=False):
        self.args = args
        if verbose:
            print(f'Running {self.bat_name} with arguments: {self.args}')
        self.cmd = self.bat_name + " " + " ".join(f'"{arg}"' for arg in args)

        p = Popen(self.cmd, stdout=PIPE, stderr=PIPE)
        output, errors = p.communicate()
        p.wait()    # wait for process to terminate

        if verbose:
            print(output)

        return p.returncode, output, errors

    def __str__(self) -> str:
        return f"""
    BatchFile(bat_name = {self.bat_name})
        Most recent args = {self.args}
        Most recent command sent to console: {self.cmd}
        """