"""

Model plotting

"""
# Function for initial plot for first inspection

def inspection(dataframe, las):
    """
    inputs: dataframe = well logging in dataframe
            las = well logging in las file
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # create figure

    fig, axis = plt.subplots(nrows = 1, ncols = len(dataframe.columns), figsize = (30,20), sharey = True)
    
    units = [curve.unit for curve in las.curves]
    index_unit = units.pop(0)

    # plot setting for all axis

    bottom_depth = dataframe.index.max()
    top_depth = dataframe.index.min()

    axis[0].set_ylabel('MD[%s]' %index_unit, fontsize = 15)

    for ax, col, unit in zip(axis, dataframe.columns, units):
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
        ax.set_xlabel(col + '\n[%s]' %unit, fontsize = 15)

        if (col == 'RT') or (col == 'MSFL'):
            ax.plot(dataframe[col], dataframe.index, linewidth = '0.5')
            ax.set_xscale('log')

        elif col == 'BHF':

            dataframe['bhf'] = np.nan
            dataframe.loc[dataframe.BHF == 'BAD', 'bhf'] = 1
            ax.fill_betweenx(dataframe.index, 0, dataframe.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
            dataframe.drop(columns = ['bhf'], inplace = True)

        elif col in ['TVD', 'TVDSS', 'AZI', 'ANG', 'CAL', 'BS']:
            ax.plot(dataframe[col], dataframe.index, linewidth = '1.0')
        
        else:
            ax.plot(dataframe[col], dataframe.index, linewidth = '0.5')

    fig.tight_layout()

    plt.show()






if __name__ == "__main__":
    
    import lasio
    import numpy as np

    path1 = '/Users/phumikrai/Documents/All Projects/1D-MEM/Saved files/LASfiles/A-1_py.las'
    path2 = '/Users/phumikrai/Documents/All Projects/1D-MEM/Sirikit field/Las files/A-1_las.las'
    
    las = lasio.read(path1)
    print(las.df())