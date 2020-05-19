try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `matplotlib` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install matplotlib`.')

from pathlib import Path


class Plotter(object):
    def __init__(self, root_path=''):
        self.root_path = Path(root_path)

    def save_current_figure(self, path, extention='.png', naked=True):
        fig, _ = plt.gcf(), plt.gca()
        # Prepare save location and check img file extention
        path = self.root_path / Path(path if str(path).endswith(extention) else f'{str(path)}{extention}')
        path.parent.mkdir(exist_ok=True, parents=True)
        if naked:
            plt.axis('off')
            fig.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0)
            fig.clf()
        else:
            fig.savefig(path)
            fig.clf()

    def show_current_figure(self):
        fig, _ = plt.gcf(), plt.gca()
        fig.show()
        fig.clf()


if __name__ == '__main__':
    output_root = Path('..') / 'output'
    p = Plotter(output_root)
    p.save_current_figure('test.png')
