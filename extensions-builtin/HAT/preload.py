import os
from modules import paths


def preload(parser):
    parser.add_argument("--hat-models-path", type=str, help="Path to directory with HAT model file(s).", default=os.path.join(paths.models_path, 'HAT'))
