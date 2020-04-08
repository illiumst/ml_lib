from pathlib import Path
import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, root_path=''):
        self.root_path = Path(root_path)

    def save_current_figure(self, path, extention='.png'):
        fig, _ = plt.gcf(), plt.gca()
        # Prepare save location and check img file extention
        path = self.root_path / Path(path if str(path).endswith(extention) else f'{str(path)}{extention}')
        path.parent.mkdir(exist_ok=True, parents=True)
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
