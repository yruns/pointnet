import os


class Args(object):

    def __init__(self):
        pass

    def print_args(self, logger):
        for attr in dir(self):
            if attr not in ['print_args', 'save_settings'] and not attr.startswith('__'):
                logger.info('{} = {}'.format(attr, getattr(self, attr)))

    def save_settings(self, output_dir):
        with open(os.path.join(output_dir, 'settings.txt'), 'w') as f:
            for attr in dir(self):
                if attr not in ['print_args', 'save_settings'] and not attr.startswith('__'):
                    f.write('{} = {}\n'.format(attr, getattr(self, attr)))

    def __repr__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    args = Args()
    args.save_settings('./logs')


