import matplotlib.pyplot as plt

def view_and_save_fig(fpath, dpi):
    fig_export = plt.gcf()
    plt.show()
    fig_export.savefig(fpath, bbox_inches='tight', dpi=dpi)