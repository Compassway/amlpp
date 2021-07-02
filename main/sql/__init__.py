import yaml
from yaml import Loader

import os

dirname = os.path.dirname(__file__)
outpath = os.path.dirname(dirname)

sql_config = yaml.load(open(os.path.join(outpath, 'config.yaml')).read(), Loader=Loader)['sql']['db']
