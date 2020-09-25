try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `matplotlib` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install matplotlib`.')

from pathlib import Path


class Plotter(object):

    def __init__(self, root_path=''):
        if not root_path:
            self.root_path = Path(root_path)

    def save_figure(self, figure, title, extention='.png', naked=False):
        canvas = FigureCanvas(figure)
        # Prepare save location and check img file extention
        path = self.root_path / f'{title}{extention}'
        path.parent.mkdir(exist_ok=True, parents=True)
        if naked:
            figure.axis('off)')
            figure.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0)
            canvas.print_figure(path)
        else:
            canvas.print_figure(path)


if __name__ == '__main__':
    raise PermissionError('Get out of here.')
