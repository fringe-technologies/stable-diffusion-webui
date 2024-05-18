import os
from modules import paths


def preload(parser):
    parser.add_argument("--span-models-path", type=str, help="Path to directory with SPAN model file(s).", default=os.path.join(paths.models_path, 'SPAN'))
