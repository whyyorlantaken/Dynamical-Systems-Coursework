# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#            Execution file for Arietta
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# This file will be used to run any of the CA models
# defined in the Arietta package.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                       Parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import argparse

def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description = "Run a cellular automaton model.")

    # and subparsers
    subparsers = parser.add_subparsers(dest = "command", required = True)

    # ===================== ELEMENTARY =====================
    elem_parser = subparsers.add_parser(
        "elementary", 
        help = "Run an elementary cellular automaton.")
    
    # Args
    elem_parser.add_argument(
        "-r", "--rule", 
        type = int, 
        required = True, 
        help = "The rule number (0-255)."
        )
    elem_parser.add_argument(
        "-l", "--length",
        type = int,
        default = 100,
        help = "The length of the cellular automaton (default: 100)."
    )
    elem_parser.add_argument(
        "-i", "--iterations",
        type = int,
        default = 100,
        help = "The number of iterations (default: 100)."
    )

    # ===================== LIFELIKE =====================
    lifelike_parser = subparsers.add_parser(
        "lifelike", 
        help = "Run a life-like cellular automaton.")
    
    # Args
    lifelike_parser.add_argument(
        "-r", "--rule", 
        type = int, 
        required = True, 
        help = "The rule number (0-262143)."
    )
    lifelike_parser.add_argument(
        "-res", "--resolution",
        type = int,
        default = 100,
        help = "The resolution of the cellular automaton (default: 100)."
    )
    lifelike_parser.add_argument(
        "-gen", "--generations",
        type = int,
        default = 100,
        help = "The number of generations (default: 100)."
    )
    lifelike_parser.add_argument(
        "-p", "--path",
        type = str,
        default = None,
        help = "The path to save the images (default: None)."
    )
    lifelike_parser.add_argument(
        "-s", "--shape",
        type = int,
        nargs = 2,
        default = (6, 6),
        help = "The shape of the images (default: (6, 6))."
    )
    lifelike_parser.add_argument(
        "-c", "--cmap",
        type = str,
        default = "Blues",
        help = "The colormap to use (default: 'Blues')."
    )
    lifelike_parser.add_argument(
        "-gn", "--gif_name",
        type = str,
        default = "evolution",
        help = "The name of the GIF file (default: 'evolution')."
    )
    lifelike_parser.add_argument(
        "-fps", "--fps",
        type = int,
        default = 5,
        help = "Frames per second for the GIF (default: 5)."
    )
    lifelike_parser.add_argument(
        "-l", "--loop",
        type = int,
        default = 0,
        help = "Loop count for the GIF (default: 0)."
    )
    return parser.parse_args()

if __name__ == "__main__":

    # Initialize the parser
    args = parse_args()

    # Package name
    print("")
    print("━━" * 30)
    print("             /\                                    ")
    print("         _  / |         .-.        /    /         ")
    print("        (  /  |  . ).--.`-' .-.---/----/---.-.    ")
    print("         `/.__|_.'/    /  ./.-'_ /    /   (  |    ")
    print("     .:' /    |  /  _.(__.(__.' /    /     `-'-'.v.1.0  ")
    print("     (__.'     `-'                                   ")
    print("━━" * 30)
    print("                CELLULAR AUTOMATA SIMULATOR")
    print("━━" * 30)

    # Elementary case
    if args.command == 'elementary':

        # Import 
        from elementary import ElementaryCA

        # Info
        print("")
        print("> Initializing ElementaryCA:")
        print(f"    Rule:        {args.rule}")
        print(f"    Seq. length: {args.length}")
        print("")

        # Create an instance
        automaton = ElementaryCA(
            rule = args.rule,
            length = args.length
        )

        # Evolve
        print(f"> Evolving cellular automaton:")
        print(f"    Iterations: {args.iterations}")
        print("")

        # Evolve
        name = f"Output/elem.r{args.rule}.{args.length}*{args.iterations}.npy"
        evolution = automaton.evolution(
            iterations = args.iterations,
            save = True,
            name = name
        )

        print(f"> Done. Evolution saved to: {name}")
        print("━━" * 30)

    # [python main.py elementary -r 30 -l 100 -i 100]

    # Lifelike case
    elif args.command == 'lifelike':

        # Import 
        from lifelike import LifeLikeCA

        # Info
        print("")
        print("> Initializing LifeLikeCA:")
        print(f"    Rule:       {args.rule}")
        print(f"    Resolution: {args.resolution}")
        print("")

        # Create an instance
        automaton = LifeLikeCA(
            rule = args.rule,
            res = args.resolution
        )

        # Generate images
        print("")
        print(f"> Generating images:")
        print(f"    Generations: {args.generations}")
        print(f"    Path:        {args.path}")
        print("")

        # Generate images
        automaton.images(
            generations = args.generations,
            path = args.path,
            shape = tuple(args.shape),
            cmap = args.cmap
        )

        # Generate GIF
        print(f"> Creating GIF:")
        print(f"    Name: {args.gif_name}")
        print(f"    FPS:  {args.fps}")

        automaton.gif(
            path = args.path,
            name = args.gif_name,
            fps = args.fps,
            loop = args.loop
        )

        print(f"> Done.")
        print("━━" * 30)

    # [python main.py lifelike -r 6152 -res 100 -gen 10 -p Output/ -gn conway]