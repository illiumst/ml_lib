try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `matplotlib` which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install matplotlib`.')

from pathlib import Path


def prettyfy_sns():
    plt.style.use('default')
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError('You want to use `seaborn` which is not installed yet,'  # pragma: no-cover
                          ' install it with `pip install seaborn`.')
    sns.set_palette('Dark2')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    plt.rcParams.update(tex_fonts)


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
