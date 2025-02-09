import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Ensure this is imported
import scipy.stats as stats
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib as mpl

# Setting color style
from .colors import COLOR_THEME, COLOR_THEME_WITH_ALPHA_SWEEP
import scienceplots
plt.style.use([ 'nature','science',"ieee",'no-latex'])

color_palette = COLOR_THEME_WITH_ALPHA_SWEEP[1:]

def add_significance(ax, data, col_name, significance_dict):
    # Add significance if provided
    if significance_dict:
        y_max = data[col_name].max()
        y_offset = y_max * 0.1  # offset for annotation in relation to the highest bar
        for comp in significance_dict.keys():
            # comp should be a tuple of the form: (bar1, bar2, height, asterisks)
            bar1, bar2, height, asterisks = comp
            x1, x2 = bar1, bar2

            #* If plotting asterisks above line
            # ax.plot([x1, x1, x2, x2], [height+y_offset, height+y_offset*1.05, height+y_offset*1.05, height+y_offset], lw=0.5, c='black')
            # ax.text((x1+x2)*0.5, height+y_offset*1.1, asterisks, ha='center', va='bottom', color='black')
            # Calculate the middle position for the asterisk and the line break
            mid_point = (x1 + x2) / 2
            asterisk_width = 0.25 # # Determine the space to leave for the asterisk * (x2 - x1)  # 10% of the bar width
            
            if asterisks != '':
                # Draw the first line segment (left)
                ax.plot([x1, x1, mid_point - asterisk_width], 
                        [height + y_offset, height + y_offset * 1.05, height + y_offset * 1.05], 
                        lw=0.5, c='black', linestyle='-')
                # Draw the second line segment (right), set line width to 0.5
                ax.plot([mid_point + asterisk_width, x2, x2], 
                        [height + y_offset * 1.05, height + y_offset * 1.05, height + y_offset], 
                        lw=0.5, c='black', linestyle='-')
                # Add asterisks in the middle of the line break
                ax.text(mid_point, height + y_offset * 1.05, asterisks, 
                        ha='center', va='center', color='black', fontsize=12, fontweight='bold') 
            
            # If asterisks are empty, plot a continuous line across
            else:
                # Draw a continuous line without breaks
                ax.plot([x1, x2], 
                        [height + y_offset, height + y_offset], 
                        lw=0.5, c='black', linestyle='-')


def barplot(save_path=None, title=None, color3_start_id=0, figsize=(10, 6),hline=None, significance_dict=None, log_scale=False, *args, **kwargs):
    with plt.style.context(['nature', 'science', "ieee", 'no-latex']):
        # Make sure to use sans-serif fonts
        fm = mpl.font_manager
        fm._get_fontconfig_fonts.cache_clear()
        plt.rcParams['font.family'] = 'sans-serif'

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(palette=COLOR_THEME_WITH_ALPHA_SWEEP[color3_start_id:], *args, **kwargs)
        ax.tick_params(right=False, top=False)
        ax.tick_params(right=False, which='minor', top=False)
        # ax.legend(title=None)
        ax.set(title=title)
        if log_scale:
            ax.set_yscale('log')

        # Set x-tick labels to have a 45 degree rotation and set error bar style to solid line
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for line in ax.lines:  # Iterate through the error bar lines
            line.set_linestyle('-')  # Set the line style to solid
            line.set_linewidth(0.5) # Set the error bar thickness to 1pt

        # Adding a horizontal line if 'hline' parameter is specified
        if hline is not None:
            plt.axhline(y=hline['value'], color=hline['color'], linestyle=hline['linestyle'], linewidth=hline['linewidth'])

        # Adding significance annotations if provided
        if significance_dict:
            add_significance(ax, kwargs.get('data'), kwargs.get('y'), significance_dict)

        sns.despine()
        if save_path:
            dir_path = os.path.dirname(save_path)
            os.mkdir(dir_path) if not os.path.exists(dir_path) else None
            plt.savefig(f'{save_path}.jpg', dpi=200)
    return ax

def plot_bar_plot_from_df(df_accuracy_ratios, path, x="Model", y="Effect", hue=None, title="Mean Accuracy Ratio Comparison", show_plot=True, color3_start_id=0, figsize=(10, 6), hline=None, significance_dict=None,log_scale=False):
    
    with plt.style.context(['nature', 'science', "ieee", 'no-latex']):
        if hue:
            ax = barplot(color3_start_id=color3_start_id, figsize=figsize, title=title, x=x, y=y, hue=hue, data=df_accuracy_ratios, edgecolor="black", linewidth=.8, capsize=.02, err_kws={'linewidth': 1.5}, hline=hline, significance_dict=significance_dict, log_scale=log_scale)
        else:
            ax = barplot(color3_start_id=color3_start_id, figsize=figsize, title=title, x=x, y=y, data=df_accuracy_ratios, edgecolor="black", linewidth=.8, capsize=.02, err_kws={'linewidth': 1.5}, hline=hline, significance_dict=significance_dict, log_scale=log_scale)
        
        if path:
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(path, dpi=200)
            print(f"Saved plot to {path}")
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_stacked_bar_plot(df_main, df_percentage, path, x="Model", y="Effect", hue="Group", title="Mean Accuracy Ratio Comparison", show_plot=True, color3_start_id=0, main_label = 'Spatial Prior', percentage_label="Common Prior"):

    from .colors import COLOR_THEME
    plt.style.use([ 'nature','science',"ieee",'no-latex'])
    with plt.style.context(['nature', 'science', "ieee", 'no-latex']):
    
        df_main['Group'] = 'Spatial Prior'
        df_percentage['Group'] = 'Common Prior'

        # Combine the two dataframes
        combined_df = pd.concat([df_main, df_percentage])

        barplot(color3_start_id=color3_start_id, title=title, x=x, y=y, hue=hue, data=combined_df, edgecolor="black", linewidth=.8, capsize=.02, errwidth=1.5)
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)

        # Rotate the x-tick labels if they overlap
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

        # Saving the plot to the specified path
        if path:
            dir_path = os.path.dirname(path)
            plt.savefig(path, dpi=200)

        # Displaying or closing the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        

def plot_mask_ditribution_img(class_name, source_dir = "/home/hpczeji1/Datasets/annotations/numpy_save/", if_show= True,if_print = False):
    """plot object distribution of one specific category (id)

    Args:
        class_name (int): category id 
        source_dir (str, optional): dir path that saved objects distribution map.npy. Defaults to "/home/hpczeji1/Datasets/annotations/numpy_save/".
        if_show (bool, optional): whether show maps. Defaults to True.
        if_print (bool, optional): if print objects distribution info. Defaults to False.

    Returns:
        matrix: object distribution map
    """    
    
    class_mask_distribution_matrix = np.load(os.path.join(source_dir,str(class_name)+".npy"))
    normed_class_mask_distribution_matrix = class_mask_distribution_matrix/np.sum(class_mask_distribution_matrix)
    
    if if_print:
        print(f"{class_name} mask_distribution_matrix shape{class_mask_distribution_matrix.shape} sum {np.sum(class_mask_distribution_matrix)} max {np.max(class_mask_distribution_matrix)} min {np.min(class_mask_distribution_matrix)} sum {np.sum(class_mask_distribution_matrix)}")
    if if_show:
        plt.imshow(normed_class_mask_distribution_matrix)

    return normed_class_mask_distribution_matrix

## --- Visualize the acc maps across different models --- ##

def plot_N_matrices_comparison(N_img_lists,model_names,N=3,N_img_name_list = None, nrows=4,ncols=4,vmin =None, vmax=None, figsize = None,show_values = "",save_path=None, vmin_max_ignore_id = None,colorbar_name='',frontsize=10,dpi=100,show_plot=False,color_platte_name="magma",*args,**kwarg):
    """Plot multi matrices togethear in comparison across 16 categories, 
        here for [Object distribution,humans, NCN, CNN ]

    Args:
        N_img_lists (list): list or Object distribution,humans, NCN, CNN accuracy matrices in 16 cateories
        N (int, optional): number of visual systems to make comparison. Defaults to 3.
        N_img_name_list (_type_, optional): name of items in  N_img_lists (list): . Defaults to None.
        nrows (int, optional): . Defaults to 4.
        ncols (int, optional): . Defaults to 4.
        vmin (_type_, optional): . Defaults to None.
        vmax (_type_, optional): . Defaults to None.
        figsize (_type_, optional): . Defaults to None.
        show_values (str, optional): . Defaults to "".
        save_path (_type_, optional): . Defaults to None.
        vmin_max_ignore_id (_type_, optional): . Defaults to None.
        colorbar_name (str, optional): . Defaults to 'KL Divergence'.
        frontsize (int, optional): . Defaults to 10.
    """
    with plt.style.context(['nature', 'science', "ieee", 'no-latex']):    
        fm = mpl.font_manager
        fm._get_fontconfig_fonts.cache_clear()
        plt.rcParams['font.family'] = 'sans-serif'

        if figsize:
            fig, axes = plt.subplots(nrows,ncols*N,figsize = figsize)
        else:
            fig, axes = plt.subplots(nrows,ncols*N)
        for i, row in enumerate(axes):
            for j, col in enumerate(row):

                index = (ncols*N*i+ j)//N
                
                for k in range(N):
                    if (N*i+ j) % N == k:
                    
                        img = N_img_lists[k][index]
                        if show_values:
                            for p in range(img.shape[0]):
                                for q in range(img.shape[1]):
                                    c = round(img[p][q],1)
                                    col.text(q+0.05, p+0.05, str(c), va='center', ha='center')

                        if vmin is not None and vmax is not None:
                            
                            if vmin_max_ignore_id ==k:
                                im = col.imshow(img, cmap=mpl.colormaps['viridis'])
                            else:
                                im = col.imshow(img,vmin= vmin, vmax=vmax, cmap=mpl.colormaps[color_platte_name])
                        else:
                            im = col.imshow(img, cmap=mpl.colormaps[color_platte_name])
                        col.set_yticks([])
                        col.set_xticks([])
                        if N_img_name_list is not None:
                            col.set_title(N_img_name_list[k][index],fontsize = frontsize) 

                        # Add model name to the last row
                        if i == nrows - 1:  # Check if it's the last row
                            col.set_xlabel(model_names[k],fontsize=frontsize+3)  # Add the model name under each subplot in the last row

                        
        position=fig.add_axes([1.02, 0.2, 0.01, 0.6])  #  [left, bottom, width, height] of colorbar position
        cb=plt.colorbar(im,cax=position,orientation='vertical') 
        cb.set_label(colorbar_name, rotation=270)
        # plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches = 'tight',facecolor="white", dpi =dpi,transparent=True)
            plt.savefig(save_path.split(".")[0]+".svg", bbox_inches = 'tight',facecolor="white", transparent=True)
        
        if show_plot:
            plt.show()
        else:
            plt.close()



def plot_rdm(rdm, percentile=False, rescale=False, lim=[0, 1], conditions=None, con_fontsize=12, cmap='magma', title=None,
             title_fontsize=12,save_path=None):
    import copy
    """
    Plot the RDM

    Parameters
    ----------
    rdm : array or list [n_cons, n_cons]
        A representational dissimilarity matrix.
    percentile : bool True or False. Default is False.
        Rescale the values in RDM or not by displaying the percentile.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    conditions : string-array or string-list. Default is None.
        The labels of the conditions for plotting.
        conditions should contain n_cons strings, If conditions=None, the labels of conditions will be invisible.
    con_fontsize : int or float. Default is 12.
        The fontsize of the labels of the conditions for plotting.
    cmap : matplotlib colormap. Default is None.
        The colormap for RDM.
        If cmap=None, the ccolormap will be 'jet'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """
    # Make sure to use sans-serif fonts
    fm = mpl.font_manager
    fm._get_fontconfig_fonts.cache_clear()
    plt.rcParams['font.family'] = 'sans-serif'

    if len(np.shape(rdm)) != 2 or np.shape(rdm)[0] != np.shape(rdm)[1]:

        return "Invalid input!"

    # get the number of conditions
    cons = rdm.shape[0]

    crdm = copy.deepcopy(rdm)

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2. Here NeuroRA cannot plot this RDM.")

        return None

    # determine if it's a square
    a, b = np.shape(crdm)
    if a != b:
        return None

    if percentile == True:

        v = np.zeros([cons * cons, 2], dtype=np.float32)
        for i in range(cons):
            for j in range(cons):
                v[i * cons + j, 0] = crdm[i, j]

        index = np.argsort(v[:, 0])
        m = 0
        for i in range(cons * cons):
            if i > 0:
                if v[index[i], 0] > v[index[i - 1], 0]:
                    m = m + 1
                v[index[i], 1] = m

        v[:, 0] = v[:, 1] * 100 / m

        for i in range(cons):
            for j in range(cons):
                crdm[i, j] = v[i * cons + j, 0]

        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(0, 100))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(0, 100))

    # rescale the RDM
    elif rescale == True:

        # flatten the RDM
        vrdm = np.reshape(rdm, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        crdm[i, j] = float((crdm[i, j] - minvalue) / (maxvalue - minvalue))

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    else:

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=5)
    font = {'size': 5}

    if percentile == True:
        cb.set_label("Dissimilarity (percentile)", fontdict=font)
    elif rescale == True:
        cb.set_label("Dissimilarity (Rescaling)", fontdict=font)
    else:
        cb.set_label("Dissimilarity", fontdict=font)

    if conditions != None:
        print("1")
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=30, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")

    plt.title(title, fontsize=title_fontsize)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    return 0

def process_and_average_data(x, y, resampling_size):
    """Process and average data for plotting."""
    x = np.mean(np.array(x), axis=0).flatten()
    y = np.mean(np.array(y).reshape(resampling_size, -1), axis=0).flatten()
    return x, y

def plot_scatter_points(ax, x, y, model_name, color, kwargs):
    """Plot scatter points on the given axis."""
    marker = kwargs.get("marker", "o")
    size_reference = kwargs.get("size_reference", None)
    if size_reference:
        sizes = [2**(5 * n) for n in size_reference]
        ax.scatter(x, y, s=sizes, marker=marker, label=model_name, color=color)
    else:
        ax.scatter(x, y, marker=marker, label=model_name, color=color)
    # Optionally add error bars
    if kwargs.get("plot_ci", True):
        add_error_bars(ax, x, y, color)

def add_error_bars(ax, x, y, color):
    """Add error bars to the plot."""
    xerr = stats.sem(x)
    yerr = stats.sem(y)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor=color, capsize=3)

def fit_and_plot_regression(ax, x, y, model_name, index, color, fit_option, kwargs):
    """Fit regression model and plot regression line with confidence interval."""
    estimator = get_estimator(fit_option)
    print(f"fit_option: {fit_option}")
    x_arr = np.array(x).reshape(-1, 1)
    estimator.fit(x_arr, y)
    plot_regression_line(ax, x_arr, y, estimator, model_name, index, color, kwargs)

def finalize_plot(ax, kwargs):
    """Finalize the plot by setting titles, labels, and layout."""
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=5)
    ax.autoscale()
    ax.set_title(kwargs.get("title", "Relationship between Variables"))
    ax.set_xlabel(kwargs.get("xlabel", 'X-axis'))
    ax.set_ylabel(kwargs.get("ylabel", 'Y-axis'))
    plt.tight_layout()

def save_plot(save_path):
    """Save the plot to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor="white", dpi=200, transparent=True)


def plot_accuracy_maps(individual_acc_maps_raw, category_names, behaviour_analysis_result_dir):
    """
    Plots accuracy maps for all categories and participants.

    Parameters:
    - individual_acc_maps_raw: np.array of shape (30, 16, 5, 5) containing accuracy data.
    - category_names: List of category names corresponding to the data.
    - behaviour_analysis_result_dir: Directory path where the result will be saved.
    """
    
    # Plotting setup
    num_cat = len(category_names)
    num_participants = individual_acc_maps_raw.shape[0]
    fig, axes = plt.subplots(nrows=num_cat, ncols=num_participants, figsize=(num_participants, num_cat), dpi=300)
    plt.subplots_adjust(hspace=0.8, wspace=0.4)

    for i in range(num_cat):  # Loop over categories
        for j in range(num_participants):  # Loop over participants
            ax = axes[i, j]
            cax = ax.matshow(individual_acc_maps_raw[j, i], cmap='magma', vmin=0, vmax=1)
            ax.axis('off')

            # Adding category names to the leftmost plot and participant numbers to the topmost plot
            if j == 0:
                ax.set_ylabel(category_names[i], rotation=0, labelpad=10, fontsize=8, va='center', ha='left')
            if i == 0:
                ax.set_title(f'P{j+1}', fontsize=8, pad=10)

    plt.tight_layout()
    fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.01, aspect=100)
    
    # Ensure the output directory exists
    maps_visualization_dir = os.path.join(behaviour_analysis_result_dir, 'maps_visualization')
    os.makedirs(maps_visualization_dir, exist_ok=True)
    plt.savefig(os.path.join(maps_visualization_dir, "30_human_acc_maps_visualization.pdf"))


def plot_with_error_bars(df, title, ylabel, model_names, size_factors, save_path, colors = COLOR_THEME_WITH_ALPHA_SWEEP[1:]):
    
    plt.figure(figsize=(3.54, 2))  # half paper width
    for model_name, color in zip(model_names, colors):
        means = df.loc[model_name].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan).values
        stds = df.loc[model_name].apply(lambda x: np.std(x) if isinstance(x, np.ndarray) else np.nan).values
        plt.errorbar(size_factors, means, yerr=stds, label=model_name, marker='o', linestyle='-', color = color)
    plt.xlabel('Object Size Factor')
    plt.ylabel(ylabel)
    plt.xticks(size_factors) # x ticks using size factors

    try:
        max_val = df.applymap(lambda x: np.max(x) if isinstance(x, np.ndarray) else np.nan).values.max()
        plt.ylim(0, max_val + 0.02) # limit y axis range from 0 to maximum value
    except:
        plt.ylim(0, 0.45)

    plt.title(title)
    plt.legend()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.tick_params(which='both', top=False, right=False)
    plt.show()
    plt.savefig(save_path)


def plot_combined_accuracy_matrices(accuracy_results, plot_save_dir):
    """
    Plots multiple accuracy matrices in a 4x4 grid and saves the combined plot.

    Parameters:
    - accuracy_results: dictionary with categories as keys and matrices as values
    - plot_save_dir: directory to save the combined plot
    """

    fig, axs = plt.subplots(4, 4, figsize=(15, 15))  # Create a 4x4 grid of subplots
    fig.tight_layout(pad=4.0)  # Add padding between subplots for better clarity

    # Iterate through each category and matrix in the accuracy results and plot them
    for index, (cat, matrix) in enumerate(accuracy_results.items()):
        row = index // 4  # Determine the row index
        col = index % 4   # Determine the column index

        #* Notice /np.max(matrix) lead matrix range in [0,1], and need to set vim = 0, vmax = 1
        im = axs[row, col].imshow(matrix/np.max(matrix), cmap='magma', interpolation='nearest',\
                                vmin =0, vmax = 1)  
        try:
            axs[row, col].set_title(f"{cat.decode('utf-8')}")
        except:
            axs[row, col].set_title(f"{cat}")

        # Only add colorbars to the rightmost plots for clarity
        if col == 3:
            fig.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)

    plot_filepath = os.path.join(plot_save_dir, "combined_accuracy_matrices.pdf")
    os.makedirs(plot_save_dir, exist_ok=True)
    plt.savefig(plot_filepath, dpi=300)  # Save with a resolution of 300 dpi


# plot and save confusion matrix
def plot_and_save_confusion_matrix(conf_matrix, categories, acc_maps_save_dir, model_name, data_id, seed, epoch):
    """
    Plot and save the confusion matrix.

    Parameters:
    - conf_matrix (ndarray): The confusion matrix to plot.
    - categories (list): List of category labels.
    - acc_maps_save_dir (str): Directory to save the confusion matrix and plot.
    - model_name (str): Name of the model.
    - data_id (str): Identifier for the dataset.
    - seed (int): Seed value.
    - epoch (int): Epoch number.
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size as needed

    # Create confusion matrix display
    # divide by the number of images per category to get the percentage
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)[:, np.newaxis]
    # keep 2 decimal places
    conf_matrix = np.round(conf_matrix, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)

    # Plot the confusion matrix
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')  # Adjust cmap and rotation as needed

    # Save the plot as an image file
    plt.savefig(f"{acc_maps_save_dir}confusion_matrix_{model_name}_{data_id}_model_id_{seed}_ep{epoch}.pdf", dpi=300)
    plt.close(fig) 