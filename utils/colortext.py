from colorama import init, Fore, Style

def init_colorama():
    init(autoreset=True)

def color_print(text: str, color: str, print_output: bool = True) -> None:
    colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE
    }

    if color not in colors:
        raise ValueError(f'Invalid color name: {color}')

    colored_text = colors[color] + text + Style.RESET_ALL

    if print_output:
        print(colored_text)
    else:
        return colored_text