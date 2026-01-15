import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_points(points:np.ndarray, labels:np.ndarray, legend=None, title=None, fig=None, nrows=1, ncols=1, index=1, colors=None, figsize=None, label_cols=10, fontsize=15):
    unique_labels = np.unique(labels)
    if not colors:
        colors = {}
        for l, c in zip(unique_labels, cm.rainbow(np.linspace(0, 1, len(unique_labels)))):
            colors[l] = c

    N_COMPONENTS = points.shape[1]

    fig_created = False

    if not fig:
        fig = plt.figure(figsize=figsize)
        fig_created = True
    
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    
    if N_COMPONENTS == 2:
        ax = fig.add_subplot(nrows, ncols, index)
    else:
        ax = fig.add_subplot(nrows, ncols, index, projection="3d")
    
    for label in unique_labels:
        points_label = points[labels == label, :]

        if N_COMPONENTS == 2:
            ax.scatter(points_label[:, 0], points_label[:, 1], color=colors[label])
        else:
            ax.scatter(points_label[:, 0], points_label[:, 1], points_label[:, 2], color=colors[label])

    if legend:
        # ax.legend([legend[label] for label in unique_labels], ncols=min(len(unique_labels), label_cols), bbox_to_anchor=(0, 1), loc='lower left', fontsize=fontsize)
        ax.legend([legend[label] for label in unique_labels], ncols=min(len(unique_labels), label_cols), fontsize=fontsize)
    else:
        # ax.legend(unique_labels, ncols=min(len(unique_labels), label_cols), bbox_to_anchor=(0, 1), loc='lower left', fontsize=fontsize)
        ax.legend(unique_labels, ncols=min(len(unique_labels), label_cols), fontsize=fontsize)
    
    if title:
        ax.set_title(title)
    
    if fig_created:
        plt.show()

    '''
    unique_labels = list(set((labels)))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    N_COMPONENTS = points.shape[1]

    if N_COMPONENTS == 3:
        # 3d
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for label, color in zip(unique_labels, colors):
            points_label = points[labels == label, :]
            ax.scatter(points_label[:, 0], points_label[:, 1], points_label[:, 2], color=color)
        
        if legend:
            fig.legend(legend)
        else:
            fig.legend(unique_labels)
    else:
        # 2d

        for label, color in zip(unique_labels, colors):
            points_label = points[labels == label, :]
            plt.scatter(points_label[:, 0], points_label[:, 1], color=color)

        if legend:
            plt.legend(legend)
        else:
            plt.legend(unique_labels)

        plt.show
    '''

def plot_points_in_multiple_subplots(points, labels, legend=None, title=None, subplot_titles=None, nrows=1, ncols=1, colors=None, label_cols=10, fontsize=15):
    fig = plt.figure()

    for row in range(nrows):
        for col in range(ncols):
            index = row * ncols + col

            if index >= len(points):
                break

            subplot_title = subplot_titles[index] if subplot_titles else None
            la = labels[index] if isinstance(labels, list) else labels
            le = legend[index] if isinstance(legend, list) else legend
            c = colors[index] if isinstance(colors, list) else colors
            
            plot_points(points[index], la, legend=le, title=subplot_title, fig=fig, nrows=nrows, ncols=ncols, index=index+1, colors=c, label_cols=label_cols, fontsize=fontsize)
    
    if title:
        fig.suptitle(title, fontsize=20)
    
    plt.show()

    '''
    unique_labels = list(set((labels)))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    for r, row in enumerate(ax):
        for c, col in enumerate(row):
            index = r * ncols + c

            col.title.set_text("Ebene " + str(index))

            for label, color in zip(unique_labels, colors):
                points_label = points[index][labels == label, :]
                col.scatter(points_label[:, 0], points_label[:, 1], color=color)

            if legend:
                col.legend(legend)
            else:
                col.legend(unique_labels)

    plt.show()
    '''

def plot_volume(dataset, subset, title=None):
    dataset_hull = ConvexHull(dataset)
    subset_hull = ConvexHull(subset)

    N_COMPONENTS = dataset.shape[1]

    fig = plt.figure()

    if title:
        fig.suptitle(title, fontsize=20)

    if N_COMPONENTS == 3:
        ax = fig.add_subplot(111, projection='3d')

        for s in dataset_hull.simplices:
            s = np.append(s, s[0])
            ax.plot(dataset[s, 0], dataset[s, 1], dataset[s, 2], color="black", alpha=1)

        for s in subset_hull.simplices:
            s = np.append(s, s[0])
            ax.plot(subset[s, 0], subset[s, 1], subset[s, 2], color="blue", alpha=1)
            
    else:
        ax = fig.add_subplot(111)

        for s in dataset_hull.simplices:
            s = np.append(s, s[0])
            ax.plot(dataset[s, 0], dataset[s, 1], color="black", alpha=1)

        for s in subset_hull.simplices:
            s = np.append(s, s[0])
            ax.plot(subset[s, 0], subset[s, 1], color="blue", alpha=1)


'''
def plot_convex_hull(points, colors, alphas):
    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point, color, alpha in zip(points, colors, alphas):
            hull = ConvexHull(point)
            for s in hull.simplices:
                s = np.append(s, s[0])
                ax.plot(point[s, 0], point[s, 1], point[s, 2], color=color, alpha=alpha)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for point, color, alpha in zip(points, colors, alphas):
            hull = ConvexHull(point)
            for s in hull.simplices:
                s = np.append(s, s[0])
                ax.plot(point[s, 0], point[s, 1], color=color, alpha=alpha)

hull_label = 0
colors = cm.rainbow(np.linspace(0, 1, len(labels)))
plot_convex_hull([images_transformed, images_transformed[image_labels == hull_label]], ["black", colors[hull_label]], alphas=[0.05, 0.5])
'''


def plot_images_in_grid(images, labels=None, title=None, nrows=None, ncols=None, size=None, pad=None):
    def plot_image(images, labels, index, subplot):
        subplot.axis("off")

        if index >= len(images):
            return
            
        if labels is not None:
            subplot.set_title(str(labels[index]))

        if len(images[index].shape) == 3 and images[index].shape[0] == 3:
            subplot.imshow(np.transpose(images[index], (1, 2, 0)))
        elif images[index].shape[0] != 3:
            subplot.imshow(images[index][0], cmap="gray")
        else:
            subplot.imshow(images[index], cmap="gray")

    if not nrows and not ncols:
        nrows = 1
        ncols = len(images)
    elif not nrows:
        nrows = int(np.ceil(len(images) / ncols))
    elif not ncols:
        ncols = int(np.ceil(len(images) / nrows))

    fig, ax = plt.subplots(nrows, ncols)

    if pad is not None:
        fig.tight_layout(pad=pad)
        fig.subplots_adjust(wspace=pad, hspace=pad)

    if size:
        fig.set_figheight(size[1])
        fig.set_figwidth(size[0])

    if nrows == 1 and ncols == 1:
        plot_image(images, labels, 0, ax)
    elif nrows == 1 or ncols == 1:
        for index, subplot in enumerate(ax):
            plot_image(images, labels, index, subplot)
    else:
        for r, row in enumerate(ax):
            for c, col in enumerate(row):
                index = r * ncols + c

                plot_image(images, labels, index, col)
                '''
                col.axis("off")

                if index >= len(images):
                    continue
                    
                if labels is not None:
                    col.set_title(str(labels[index]))

                if len(images[index].shape) == 3:
                    col.imshow(np.transpose(images[index], (1, 2, 0)))
                else:
                    col.imshow(images[index], cmap="gray")
                '''
    
    if title:
        fig.suptitle(title, fontsize=20)

    plt.show()
