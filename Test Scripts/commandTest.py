import sys

def handleStartUpCommands(help_message):
    argm = []
    for arg in sys.argv[1:]:
        argm.append(arg)

    if argm[0] == 'help':
        print(help_message)
        quit()

    return argm

help_message = """This is a test script"""

def main():
    argm = handleStartUpCommands(help_message)

    for arg in argm:
        if '\\' in arg or '/' in arg:
            print(f"PATH: {arg}")
        else:
            print(f"ARG: {arg}")

if __name__ == '__main__':
    main()
