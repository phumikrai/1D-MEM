"""

Model plots

"""

# function for a log plot

from mem.audit import filename


def logplot(**kwargs):
    """
    input:  axis = sub-axis,
            log = log data,
            depth = depth data,
            color = log color,
            label = log label,
            position = lebel and tick position,
            scale = plot scale,
            scalelabel = list of label scale,
            logarithm = True for logarithm scale plot,
            style = line plot style (default: solid line),
            width = line width (default: 0.5)      
    """

    axis = kwargs.get('axis')
    log = kwargs.get('log')
    depth = kwargs.get('depth')
    color = kwargs.get('color')
    label = kwargs.get('label')
    position = kwargs.get('position')
    scale = kwargs.get('scale')
    scalelabel = kwargs.get('scalelabel')

    if 'logarithm' in kwargs:
        if kwargs.get('logarithm'):
            axis.set_xscale('log')

    if 'style' in kwargs:
        style = kwargs.get('style')
    else:
        style = '-'

    if 'width' in kwargs:
        width = kwargs.get('width')
    else:
        width = '0.5'
    
    axis.plot(log, depth, color=color, linewidth=width, linestyle=style)
    axis.spines['top'].set_position(('axes', position))
    axis.spines['top'].set_edgecolor(color)
    axis.set_xlim(scale[0], scale[-1])
    axis.set_xlabel(label, color=color)
    axis.tick_params(axis='x', colors=color)
    axis.set_xticks(scale)
    axis.set_xticklabels(scalelabel)

    # tick alignment

    xticks = axis.xaxis.get_major_ticks()
    xticks[0].label2.set_horizontalalignment('left')   # left align first tick 
    xticks[-1].label2.set_horizontalalignment('right') # right align last tick

# function for getting plot scale of pressure plot

def plotscale(**kwargs):
    """
    input:  maxscale = maximum value,
            minscale = minimum value,
            increment = increment value
    ...
    output: scale and list of scale plot
    """

    maxscale = kwargs.get('maxscale')
    minscale = kwargs.get('minscale')
    increment = kwargs.get('increment')

    import numpy as np

    # setup pressure plot scale

    x = increment
    start = np.ceil(float(minscale)/increment) * increment
    stop = (np.ceil(float(maxscale)/increment) * increment) + (increment * 1.1)
    scale = np.arange(start, stop, increment)
    
    while len(scale) > 8:
        x += increment
        scale = np.arange(start, stop, x)

    scalelist = [str(int(start))]
    for i in range(len(scale) - 2):
        scalelist.append('')
    scalelist.append(str(int(scale[-1])))

    return scale, scalelist

# function for 1D MEM plot

def mem_plot(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            formtop = formation top data (.csv),
            pres = pressure data (.csv),
            core = core data (.csv),
            forms = formation name and its color code in dictonary,
            toprange = top depth point,
            botrange = bottom depth point,
            wellname = well name,
            savepath = path to save folder
    """

    dataframe = kwargs.get('dataframe')
    formtop = kwargs.get('formtop')
    pres = kwargs.get('pres')
    core = kwargs.get('core')
    forms = kwargs.get('forms')
    toprange = kwargs.get('toprange')
    botrange = kwargs.get('botrange')
    wellname = kwargs.get('wellname')
    savepath = kwargs.get('savepath')

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as pe

    # Create figure

    fig = plt.figure(figsize=(30, 15))
    fig.suptitle('1D MEM of well %s' %wellname, fontsize=20, y=1.07)

    w_ratio = [1] + [2 for i in range(9)] + [4, 6] + [2 for i in range(4)]
    gs = gridspec.GridSpec(ncols = 16, nrows = 1, width_ratios = w_ratio)

    axis0 = fig.add_subplot(gs[0])
    axis = [axis0] + [fig.add_subplot(gs[i+1], sharey=axis0) for i in range(15)]
    fig.subplots_adjust(wspace=0.05)

    # General setting for all axis

    axis[0].set_ylabel('TVDSS[m]')

    condition = (dataframe.index >= toprange) & (dataframe.index <= botrange)
    topdepth = dataframe.loc[condition, 'TVDSS'].dropna().min()
    botdepth = dataframe.loc[condition, 'TVDSS'].dropna().max()
    
    for ax in axis:
        ax.set_ylim(topdepth, botdepth)
        ax.invert_yaxis()
        ax.minorticks_on() # scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')

    for ax in axis[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # formations plot

    ax11 = axis[0].twiny()
    ax11.set_xlim(0, 1)
    ax11.set_xticks([0, 1])
    ax11.set_xticklabels(['', ''])

    for top, bot, form in zip(formtop.TVDSS_TOP, formtop.TVDSS_BOT, formtop.FORM):
        if (top >= topdepth) & (top <= botdepth):
            ax11.axhline(y=top, linewidth=1.5, color=forms[form], alpha=0.5)
            if (bot <= botdepth):
                middle = top + (bot - top)/2
                ax11.axhspan(top, bot, facecolor=forms[form], alpha=0.2)      
            else:
                middle = top + (botdepth - top)/2
                ax11.axhspan(top, botdepth, facecolor=forms[form], alpha=0.2)
            ax11.text(0.5, middle, form, ha='center', va='center', color=forms[form],
                        path_effects = [pe.withStroke(linewidth=3, foreground="white")], fontsize=10, weight='bold')

    ax11.grid(False)

    # formation plots for other columns

    for ax in axis[1:-2]:
        for top, bot, form in zip(formtop.TVDSS_TOP, formtop.TVDSS_BOT, formtop.FORM):
            if (top >= topdepth) & (top <= botdepth):
                ax.axhline(y=top, linewidth=1.5, color=forms[form], alpha=0.5)
                ax.axhspan(top, bot, facecolor=forms[form], alpha=0.2)

    # Azimuth and angle plots
    
    ax21 = axis[1].twiny()
    logplot(axis=ax21, log=dataframe.AZI, depth=dataframe.TVDSS, color='blue', label='AZIMUTH[degree]',
                position=1.02, scale=np.arange(0, 361, 90), scalelabel=['0', '', '180', '', '360'],
                width='0.8')
    ax21.grid(True)
    
    ax22 = axis[1].twiny()
    logplot(axis=ax22, log=dataframe.ANG, depth=dataframe.TVDSS, color='red', label='ANGLE[degree]',
                position=1.06, scale=np.arange(0, 91, 45), scalelabel=['0', '45', '90'],
                width='0.8')

    # Gamma ray plot
    
    ax31 = axis[2].twiny()
    logplot(axis=ax31, log=dataframe.GR_NORM, depth=dataframe.TVDSS, color='green', label='GR[API]',
                position=1.02, scale=np.arange(0, 151, 30), scalelabel=['0', '', '', '', '','150'])
    ax31.grid(True)
    
    # Resisitivity plots
    
    ax41 = axis[3].twiny()
    logplot(axis=ax41, log=dataframe.RT, depth=dataframe.TVDSS, color='red', label='RT[ohm-m]',
                position=1.02, scale=[0.2, 2, 20, 200, 2000], scalelabel=['0.2', '', '', '', '2000'],
                logarithm=True)
    ax41.grid(True)

    ax42 = axis[3].twiny()
    logplot(axis=ax42, log=dataframe.MSFL, depth=dataframe.TVDSS, color='black', label='MSFL[ohm-m]',
                position=1.06, scale=[0.2, 2, 20, 200, 2000], scalelabel=['0.2', '', '', '', '2000'],
                logarithm=True)
    
    # Density and neutron porosity plots
    
    ax51 = axis[4].twiny()
    logplot(axis=ax51, log=dataframe.RHOB_MRG, depth=dataframe.TVDSS, color='red', label='RHOB_MRG[g/c3]',
                position=1.02, scale=np.arange(1.95, 2.96, 0.2), scalelabel=['1.95', '', '', '', '', '2.95'])
    ax51.grid(True)
    
    ax52 = axis[4].twiny()
    logplot(axis=ax52, log=dataframe.NPHI_MRG, depth=dataframe.TVDSS, color='blue', label='NPHI_MRG[V/V]',
                position=1.06, scale=np.arange(0.45, -0.16, -0.12), scalelabel=['0.45', '', '', '', '', '-0.15'])
    
    # P_Sonic and S_Sonic plots
    
    ax61 = axis[5].twiny()
    logplot(axis=ax61, log=dataframe.DTC_MRG, depth=dataframe.TVDSS, color='blue', label='DTC_MRG[us/ft]',
                position=1.02, scale=np.arange(140, 39, -20), scalelabel=['140', '', '', '', '', '40'])
    ax61.grid(True)

    ax62 = axis[5].twiny()
    logplot(axis=ax62, log=dataframe.DTS_MRG, depth=dataframe.TVDSS, color='red', label='DTS_MRG[us/ft]',
                position=1.06, scale=np.arange(340, 39, -60), scalelabel=['340', '', '', '', '', '40'])

    # Young's modulus plot

    yme = dataframe.YME/1e6
    yme_core = core.YME/1e6
    scale1, scalelist1 = plotscale(maxscale=yme.max(), minscale=0, increment=1)
    
    ax71 = axis[6].twiny()
    logplot(axis=ax71, log=yme, depth=dataframe.TVDSS, color='chocolate', label='YME[Mpsi]',
                position=1.02, scale=scale1, scalelabel=scalelist1)
    ax71.grid(True)

    ax72 = axis[6].twiny()
    ax72.scatter(yme_core, core.TVDSS, c='black', alpha=1, marker='o')
    ax72.get_xaxis().set_visible(False) 
    ax72.set_xlim(scale1[0], scale1[-1])
    
    # Poisson's ratio plot
    
    ax81 = axis[7].twiny()
    logplot(axis=ax81, log=dataframe.PR, depth=dataframe.TVDSS, color='salmon', label='PR[unitless]',
                position=1.02, scale=np.arange(0, 0.51, 0.1), scalelabel=['0', '', '', '', '','0.5'])
    ax81.grid(True)

    # unconfined compressive strength and Tensile formation strength plots

    maxscale1 = max([dataframe.UCS.max(), dataframe.TSTR.max()])
    scale2, scalelist2 = plotscale(maxscale=maxscale1, minscale=0, increment=2000)
    
    ax91 = axis[8].twiny()
    logplot(axis=ax91, log=dataframe.UCS, depth=dataframe.TVDSS, color='red', label='UCS[psi]',
                position=1.02, scale=scale2, scalelabel=scalelist2)
    ax91.grid(True)

    ax92 = axis[8].twiny()
    logplot(axis=ax92, log=dataframe.TSTR, depth=dataframe.TVDSS, color='darkorange', label='TSTR[psi]',
                position=1.06, scale=scale2, scalelabel=scalelist2)

    ax93 = axis[8].twiny() 
    ax93.scatter(core.UCS, core.TVDSS, c='black', alpha=1, marker='o')
    ax93.get_xaxis().set_visible(False) 
    ax93.set_xlim(scale2[0], scale2[-1])
    
    # angle of internal friction plot
    
    ax101 = axis[9].twiny()
    logplot(axis=ax101, log=dataframe.FANG, depth=dataframe.TVDSS, color='green', label='FANG[degree]',
                position=1.02, scale=np.arange(0, 51, 10), scalelabel=['0', '', '', '', '','50'])    
    ax101.grid(True)
    
    # principle stresses plots

    maxscale2 = max([dataframe.OBP.max(), dataframe.SHmax.max()])
    scale3, scalelist3 = plotscale(maxscale=maxscale2, minscale=0, increment=2000)
    mudpres = (dataframe.MW * 0.052) * (dataframe.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    ax111 = axis[10].twiny()
    logplot(axis=ax111, log=dataframe.SHmax, depth=dataframe.TVDSS, color='blue', label='SHmax[psi]',
                position=1.02, scale=scale3, scalelabel=scalelist3)
    ax111.grid(True)
    
    ax112 = axis[10].twiny()
    logplot(axis=ax112, log=dataframe.Shmin, depth=dataframe.TVDSS, color='lime', label='Shmin[psi]',
                position=1.06, scale=scale3, scalelabel=scalelist3)

    ax113 = axis[10].twiny()
    logplot(axis=ax113, log=dataframe.OBP, depth=dataframe.TVDSS, color='black', label='OBP[psi]',
                position=1.10, scale=scale3, scalelabel=scalelist3)

    ax114 = axis[10].twiny()
    logplot(axis=ax114, log=dataframe.PP, depth=dataframe.TVDSS, color='deepskyblue', label='PP[psi]',
                position=1.14, scale=scale3, scalelabel=scalelist3)
    
    ax115 = axis[10].twiny()
    logplot(axis=ax115, log=mudpres, depth=dataframe.TVDSS, color='green', label='MW[psi]',
                position=1.18, scale=scale3, scalelabel=scalelist3, style='--', width='1')

    ax116 = axis[10].twiny() 
    ax116.scatter(pres.PP, pres.TVDSS, c='black', alpha=1, marker='o')
    ax116.get_xaxis().set_visible(False) 
    ax116.set_xlim(scale3[0], scale3[-1])
    
    # mud window plots

    scale4 = np.arange(8, 19, 2)
    scalelist4 = ['8', '', '', '', '', '18']

    ax121 = axis[11].twiny()
    logplot(axis=ax121, log=dataframe.CMW_KICK, depth=dataframe.TVDSS, color='gray', label='CMW_KICK[ppg]',
                position=1.02, scale=scale4, scalelabel=scalelist4)
    ax121.grid(True)
    
    ax122 = axis[11].twiny()
    logplot(axis=ax122, log=dataframe.CMW_MIN_MC, depth=dataframe.TVDSS, color='red', label='CMW_MIN_MC[ppg]',
                position=1.06, scale=scale4, scalelabel=scalelist4)

    ax123 = axis[11].twiny()
    logplot(axis=ax123, log=dataframe.CMW_LOSS, depth=dataframe.TVDSS, color='indigo', label='CMW_LOSS[ppg]',
                position=1.10, scale=scale4, scalelabel=scalelist4)

    ax124 = axis[11].twiny()
    logplot(axis=ax124, log=dataframe.CMW_MAX_MTS, depth=dataframe.TVDSS, color='darkslateblue', label='CMW_MAX_MTS[ppg]',
                position=1.14, scale=scale4, scalelabel=scalelist4)
    
    ax125 = axis[11].twiny()
    logplot(axis=ax125, log=dataframe.MW, depth=dataframe.TVDSS, color='green', label='MW[ppg]',
                position=1.18, scale=scale4, scalelabel=scalelist4, style='--', width='1')

    loc1 = dataframe.CMW_KICK > scale4[0]
    loc2 = dataframe.CMW_MIN_MC > dataframe.CMW_KICK
    loc3 = dataframe.CMW_MAX_MTS > dataframe.CMW_LOSS
    loc4 = scale4[-1] > dataframe.CMW_MAX_MTS
    
    ax126 = axis[11].twiny()
    ax126.set_xlim(scale4[0], scale4[-1])
    ax126.fill_betweenx(dataframe.TVDSS, scale4[0], dataframe.CMW_KICK, where=loc1, color='silver', 
                            capstyle='butt', linewidth=0.5, alpha=0.5, label='KICK')
    ax126.fill_betweenx(dataframe.TVDSS, dataframe.CMW_KICK, dataframe.CMW_MIN_MC, where=loc2, color='yellow', 
                            capstyle='butt', linewidth=0.5, alpha=0.5, label='BREAKOUT')
    ax126.fill_betweenx(dataframe.TVDSS, dataframe.CMW_LOSS, dataframe.CMW_MAX_MTS, where=loc3, color='slateblue', 
                            capstyle='butt', linewidth=0.5, alpha=0.5, label='LOSS')
    ax126.fill_betweenx(dataframe.TVDSS, dataframe.CMW_MAX_MTS, scale4[-1], where=loc4, color='darkslateblue', 
                            capstyle='butt', linewidth=0.5, alpha=0.5, label='BREAKDOWN')
    ax126.set_xticks(scale4)
    ax126.set_xticklabels(['', '', '', '', '', ''])
    ax126.legend(loc = 'upper left')
    
    # breakout width plot

    wbo1 = dataframe.WBO / 2
    wbo2 = (dataframe.WBO / 2) * (-1)

    ax131 = axis[12].twiny()
    ax131.fill_betweenx(dataframe.TVDSS, wbo2, wbo1, color='red', capstyle='butt', linewidth=1, label='BAD')
    ax131.spines['top'].set_position(('axes', 1.02))
    ax131.spines['top'].set_edgecolor('red')   
    ax131.set_xlim(-90, 90)
    ax131.set_xlabel('WBO[degree]', color = 'red')    
    ax131.tick_params(axis = 'x', colors = 'red')
    ax131.set_xticks(np.arange(-90, 91, 45))
    ax131.set_xticklabels(['-90', '', '', '', '90'])
    ax131.grid(True)

    ax131_ticks = ax131.xaxis.get_major_ticks()
    ax131_ticks[0].label2.set_horizontalalignment('left')   # left align first tick 
    ax131_ticks[-1].label2.set_horizontalalignment('right') # right align last tick

    # caliper and bitsize plots

    scale5 = np.arange(6, 12, 1)
    scalelist5 = ['6', '', '', '', '', '11']
    
    ax141 = axis[13].twiny()
    logplot(axis=ax141, log=dataframe.BS, depth=dataframe.TVDSS, color='black', label='BS[in]',
                position=1.02, scale=scale5, scalelabel=scalelist5)
    ax141.grid(True)
    
    ax142 = axis[13].twiny()
    logplot(axis=ax142, log=dataframe.CAL, depth=dataframe.TVDSS, color='grey', label='CAL[in]',
                position=1.06, scale=scale5, scalelabel=scalelist5)

    loc5 = dataframe.BS > dataframe.CAL
    loc6 = dataframe.CAL > dataframe.BS

    ax143 = axis[13].twiny()
    ax143.set_xlim(scale5[0], scale5[-1])
    ax143.fill_betweenx(dataframe.TVDSS, dataframe.CAL, dataframe.BS, where=loc5, color='yellow', 
                            capstyle='butt', linewidth=0.5, alpha=0.5)
    ax143.fill_betweenx(dataframe.TVDSS, dataframe.BS, dataframe.CAL, where=loc6, color='red', 
                            capstyle='butt', linewidth=0.5, alpha=0.5)
    ax143.set_xticks(scale5)
    ax143.set_xticklabels(['', '', '', '', '', ''])

    # effective porosity, rock matrix, volume of clay plots

    ax151 = axis[14].twiny()
    logplot(axis=ax151, log=dataframe.VCL, depth=dataframe.TVDSS, color='SaddleBrown', label='VCL[V/V]',
                position=1.02, scale=np.arange(0, 1.1, 0.2), scalelabel=['0', '', '', '', '','1'])

    ax152 = axis[14].twiny()
    logplot(axis=ax152, log=dataframe.PHIE, depth=dataframe.TVDSS, color='gray', label='PHIE[V/V]',
                position=1.06, scale=np.arange(1.0, -0.1, -0.2), scalelabel=['1', '', '', '', '','0'])

    ax153 = axis[14].twiny()
    ax153.set_xlim(0, 1)
    ax153.fill_betweenx(dataframe.TVDSS, 0, dataframe.VCL, color='SaddleBrown', 
                            capstyle='butt', linewidth=0.5)
    ax153.fill_betweenx(dataframe.TVDSS, dataframe.VCL, (1 - dataframe.PHIE), color='yellow', 
                            capstyle='butt', linewidth=0.5)
    ax153.fill_betweenx(dataframe.TVDSS, (1 - dataframe.PHIE), 1, color='gray', 
                            capstyle='butt', linewidth=0.5)
    ax153.set_xticks([0, 1])
    ax153.set_xticklabels(['', ''])

    # plot sand-shale lithology

    dataframe['liplot'] = np.nan
    dataframe.loc[dataframe.LITHO == 'SAND', 'liplot'] = 1
    dataframe.loc[dataframe.LITHO == 'SHALE', 'liplot'] = 0
        
    ax161 = axis[15].twiny()
    ax161.fill_betweenx(dataframe.TVDSS, dataframe.liplot, 1, color='SaddleBrown', 
                            capstyle='butt', linewidth=0.01, label='SHALE')
    ax161.fill_betweenx(dataframe.TVDSS, 0, dataframe.liplot, color='yellow', 
                            capstyle='butt', linewidth=0.01, label='SAND')
    ax161.spines['top'].set_position(('axes', 1.02))
    ax161.spines['top'].set_edgecolor('gray')
    ax161.set_xlim(0, 1)
    ax161.set_xlabel('LITHOLOGY', color = 'gray')
    ax161.tick_params(axis = 'x', colors = 'gray')
    ax161.set_xticks([0, 1])
    ax161.set_xticklabels(['', ''])
    ax161.legend(loc = 'upper left')

    dataframe.drop(columns = ['liplot'], inplace = True)

    # Save files

    filename = '%s_MEM.jpg' %wellname
    memfolder = 'MEM'
    mempath = os.path.join(savepath, memfolder)

    if not os.path.isdir(mempath):
        os.makedirs(mempath)

    plt.savefig(os.path.join(mempath, filename), dpi=300, format='jpg', bbox_inches="tight")
    plt.close(fig)

# Function for check the quality of the input data for interested zone

def boxes_plot(**kwargs):
    """
    input:  dataframes = list of well logging data in dataframe,
            formtops = formation top data (.csv),
            wellids = dictionary of name and color code,
            formation = name of the formation where the data can be compared,
            savepath = path to save folder
    """

    dataframes = kwargs.get('dataframes')
    formtops = kwargs.get('formtops')
    wellids = kwargs.get('wellids')
    formation = kwargs.get('formation')
    savepath = kwargs.get('savepath')

    import os
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Set data for specific interval
    
    GR_plot, RHOB_plot, NPHI_plot, DTC_plot, DTS_plot = [list() for i in range(5)]
    YME_plot, PR_plot, UCS_plot, FANG_plot, TSTR_plot = [list() for i in range(5)]

    data_plots = [GR_plot, RHOB_plot, NPHI_plot, DTC_plot, DTS_plot,
                  YME_plot, PR_plot, UCS_plot, FANG_plot, TSTR_plot]
    
    selected_cols = ['GR_NORM', 'RHOB_MRG', 'NPHI_MRG', 'DTC_MRG', 
                     'DTS_MRG','YME', 'PR', 'UCS', 'FANG', 'TSTR']
    
    curve_labels = ['GR', 'RHOB', 'NPHI', 'DTC', 'DTS',
                    'YME', 'PR', 'UCS', 'FANG', 'TSTR']

    unit_labels = ['API', 'g/c3', 'V/V', 'us/ft', 'us/ft',
                    'psi', 'unitless', 'psi', 'degree', 'psi']

    welllabels = []
    
    for dataframe, top, name in zip(dataframes, formtops, wellids):
        
        # Check available data for selected formation
        
        if formation in list(top.FORM):
            
            # Set interval from each well for selected formation

            topdepth = float(top.loc[top.FORM == formation, 'TOP'])
            botdepth = float(top.loc[top.FORM == formation, 'BOT'])

            welllabels.append(name)
            
            # Select data from each well by interval

            condition = (dataframe.index >= topdepth) & (dataframe.index <= botdepth)

            for store, col in zip(data_plots, selected_cols):
                store.append(dataframe.loc[condition, col].dropna())

    # Setup well colors for plotting

    wellcolors = []

    for name in welllabels:
        wellcolors.append(wellids[name])

    well_colo = [item for sublist in [(c, c) for c in wellcolors] for item in sublist]
    
    # Create figure
    
    fig, axis = plt.subplots(nrows=2, ncols=5, figsize=(12.5, 5), sharey=False)
    fig.suptitle('Boxes Plot of formation %s' %formation, fontsize=20, y=1.0)
    axis = axis.flatten()
    
    # Plot setting for all axis
    
    for data, label, unit, ax in zip(data_plots, curve_labels, unit_labels, axis):
        boxes = ax.boxplot(data, labels=welllabels, meanline=True, notch=True, showfliers=False, patch_artist=True)
        ax.set_ylabel('[%s]' %unit)

        # scientific notation for YME

        if label == 'YME':
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True) 
            formatter.set_powerlimits((-5,5))
            ax.yaxis.set_major_formatter(formatter)
        
        # set decoration
        for patch, color in zip(boxes['boxes'], wellcolors): 
            patch.set_facecolor(color) 
        
        for box_wk, box_cap, color in zip(boxes['whiskers'], boxes['caps'], well_colo):
            box_wk.set(color=color, linewidth=1.5)
            box_cap.set(color=color, linewidth=3)
        
        for median in boxes['medians']:
            median.set(color = 'black', linewidth=3)  
        ax.set_title(label)
    
    fig.tight_layout()

    # Save files

    filename = 'LQC_%s_Boxplot.jpg' %formation
    LQCfolder = 'LQC'
    LQCpath = os.path.join(savepath, LQCfolder)

    if not os.path.isdir(LQCpath):
        os.makedirs(LQCpath)

    plt.savefig(os.path.join(LQCpath, filename), dpi=300, format='jpg', bbox_inches="tight")
    plt.close(fig)

# function for overburden stress plot

def obp_plot(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            wellname = well name,
            savepath = path to save folder
    """

    dataframe = kwargs.get('dataframe')
    wellname = kwargs.get('wellname')
    savepath = kwargs.get('savepath')

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 12), sharey = True)
    fig.suptitle('Overburden Pressure of Well %s' %wellname, fontsize= 20, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('TVD[m]')
    
    for ax in axis:
        ax.set_ylim(0, dataframe.TVD.max())
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')
    
    # Density and extropolated density

    scale1 = np.arange(0, 3.1, 0.5)
    scalelist1 = ['0', '', '', '', '', '', '3']
    
    ax11 = axis[0].twiny()
    logplot(axis=ax11, log=dataframe.RHOB_MRG, depth=dataframe.TVD, color='red', label='RHOB[g/c3]',
                position=1.02, scale=scale1, scalelabel=scalelist1)    
    ax11.grid(True)

    ax12 = axis[0].twiny()
    logplot(axis=ax12, log=dataframe.RHOB_EX, depth=dataframe.TVD, color='black', label='RHOB_EX[g/c3]',
                position=1.08, scale=scale1, scalelabel=scalelist1, width='1')
    
    # Overdurden Pressure

    scale2, scalelist2 = plotscale(maxscale=dataframe.OBP.max(), minscale=0, increment=2000)
    
    ax21 = axis[1].twiny()
    logplot(axis=ax21, log=dataframe.OBP, depth=dataframe.TVD, color='black', label='OBP[psi]',
                position=1.02, scale=scale2, scalelabel=scalelist2, width='1')  
    ax21.grid(True)

    # Overdurden Gradient

    scale3, scalelist3 = plotscale(maxscale=dataframe.OBG.max(), minscale=0, increment=2)

    ax31 = axis[2].twiny()
    logplot(axis=ax31, log=dataframe.OBG, depth=dataframe.TVD, color='black', label='OBG[ppg]',
                position=1.02, scale=scale3, scalelabel=scalelist3, width='1')  
    ax31.grid(True)
        
    fig.tight_layout()

    # Save files

    filename = '%s_OBP.jpg' %wellname
    obpfolder = 'OBP'
    obppath = os.path.join(savepath, obpfolder)

    if not os.path.isdir(obppath):
        os.makedirs(obppath)

    plt.savefig(os.path.join(obppath, filename), dpi=300, format='jpg', bbox_inches="tight")
    plt.close(fig)

# function for gamma ray normalization

def grnorm_plot(**kwargs):
    """
    input:  dataframes = list of well logging data in dataframe,
            wellids = dictionary of name and color code,
            savepath = path to save folder
    """

    dataframes = kwargs.get('dataframes')
    wellids = kwargs.get('wellids')
    savepath = kwargs.get('savepath')

    import os
    import matplotlib.pyplot as plt

    # bins = a number of histogram bar (default = 150)

    bins = 150

    # create figure
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6), sharey = True)
    fig.suptitle('Histogram of Normalized Gamma Ray', fontsize= 15, y = 0.98)
    
    # plot histrogram
    for dataframe, name in zip(dataframes, wellids):
        ax[0].hist(dataframe.GR, bins=bins, histtype='step', label=name, color=wellids[name])
        ax[0].set_xlabel('Gamma Ray [API]')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Before')
        ax[0].legend(loc='upper left')
        
        ax[1].hist(dataframe.GR_NORM, bins=bins, histtype='step', label=name, color=wellids[name])
        ax[1].set_xlabel('Normalized Gamma Ray [API]')
        ax[1].set_title('After')
        ax[1].legend(loc='upper left')

    fig.tight_layout()

    # save files

    filename = 'LQC_GR_NORM.jpg'
    LQCfolder = 'LQC'
    LQCpath = os.path.join(savepath, LQCfolder)

    if not os.path.isdir(LQCpath):
        os.makedirs(LQCpath)

    plt.savefig(os.path.join(LQCpath, filename), dpi=300, format='jpg', bbox_inches='tight')
    plt.close(fig)

# function for neutron-density crossplot

def nd_plot(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            formtop = formation top data (.csv),
            forms = formation name and its color code in dictonary,
            toprange = top depth point,
            botrange = bottom depth point,
            wellname = well name
            savepath = path to save folder
    """

    dataframe = kwargs.get('dataframe')
    formtop = kwargs.get('formtop')
    forms = kwargs.get('forms')
    toprange = kwargs.get('toprange')
    botrange = kwargs.get('botrange')
    wellname = kwargs.get('wellname')
    savepath = kwargs.get('savepath')

    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as pe

    # input parameters

    condition1 = (dataframe.index >= toprange) & (dataframe.index <= botrange)
    
    plotdata = dataframe.loc[condition1, ['TVDSS', 'GR_NORM', 'NPHI_MRG', 'RHOB_MRG', 'VCL', 'PHIE', 'LITHO']]
    topdepth = dataframe.loc[condition1, 'TVDSS'].dropna().min()
    botdepth = dataframe.loc[condition1, 'TVDSS'].dropna().max()

    # Create figure

    fig = plt.figure(figsize=(16, 8))

    gs = gridspec.GridSpec(ncols=4, nrows=1, width_ratios = [5, 1, 1, 1])
    axis1 = fig.add_subplot(gs[0])
    axis2 = fig.add_subplot(gs[1])
    axis3 = fig.add_subplot(gs[2], sharey=axis2)
    axis4 = fig.add_subplot(gs[3], sharey=axis2)

    # plot neutron porosity and density

    q1 = plotdata.GR_NORM.quantile(0.01)
    q99 = plotdata.GR_NORM.quantile(0.99)
    condition2 = (plotdata.GR_NORM >= q1) & (plotdata.GR_NORM <= q99)
    
    NPHI = plotdata.loc[condition2, 'NPHI_MRG']
    RHOB = plotdata.loc[condition2, 'RHOB_MRG']
    GR = plotdata.loc[condition2, 'GR_NORM']

    cmap = mpl.cm.jet

    im = axis1.scatter(NPHI, RHOB, c=GR, marker='.', cmap=cmap)

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    x_line = [0.45, 0, NPHIsh]
    y_line = [1.9, 2.65, RHOBsh]
    minerals = ['Water', 'Quartz', 'Clay']

    line = mlines.Line2D(x_line, y_line, color = 'black')
    axis1.add_line(line)

    for x, y, mineral in zip(x_line, y_line, minerals):
        axis1.text(x + 0.02, y + 0.02, mineral, ha='center', va='center', color='black',
                    path_effects = [pe.withStroke(linewidth=3, foreground="white")], fontsize=10, weight='bold')

    axis1.set_xlabel('NPHI[V/V]')
    axis1.set_ylabel('RHOB[g/c3]')
    axis1.set_xlim(-.05, .50)
    axis1.set_ylim(3, 1.8)
    axis1.grid(True)
    
    cbar = fig.colorbar(im, ax = axis1)
    cbar.set_label('NORMALIZED GAMMA RAY')

    axis1.set_title('Neutron-density crossplot of Well %s' %wellname, fontsize = 15, y = 1.06)

    # general setting for log plot

    axis2.set_ylabel('TVDSS[m]')

    for ax in [axis2, axis3, axis4]:
        ax.set_ylim(topdepth, botdepth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 

    for ax in [axis3, axis4]:
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    # formations plot

    ax21 = axis2.twiny()
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('black')
    ax21.set_xlim(0, 1)
    ax21.set_xlabel('FORMATIONS', color = 'black')    
    ax21.set_xticks([0, 1])
    ax21.set_xticklabels(['', ''])

    for top, bot, form in zip(formtop.TVDSS_TOP, formtop.TVDSS_BOT, formtop.FORM):
        if (top >= topdepth) & (top <= botdepth):
            ax21.axhline(y=top, linewidth=1.5, color=forms[form], alpha=0.5)
            if (bot <= botdepth):
                middle = top + (bot - top)/2
                ax21.axhspan(top, bot, facecolor=forms[form], alpha=0.2)      
            else:
                middle = top + (botdepth - top)/2
                ax21.axhspan(top, botdepth, facecolor=forms[form], alpha=0.2)
            ax21.text(0.5, middle, form, ha='center', va='center', color=forms[form],
                        path_effects = [pe.withStroke(linewidth=3, foreground="white")], fontsize=10, weight='bold')
    
    ax21.grid(False)

    # effective porosity, rock matrix, volume of clay plots

    ax31 = axis3.twiny()
    logplot(axis=ax31, log=plotdata.VCL, depth=plotdata.TVDSS, color='SaddleBrown', label='VCL[V/V]',
                position=1.02, scale=np.arange(0, 1.1, 0.2), scalelabel=['0', '', '', '', '','1'])

    ax32 = axis3.twiny()
    logplot(axis=ax32, log=plotdata.PHIE, depth=plotdata.TVDSS, color='gray', label='PHIE[V/V]',
                position=1.10, scale=np.arange(1.0, -0.1, -0.2), scalelabel=['1', '', '', '', '','0'])

    ax33 = axis3.twiny()
    ax33.set_xlim(0, 1)
    ax33.fill_betweenx(plotdata.TVDSS, 0, plotdata.VCL, color='SaddleBrown', 
                            capstyle='butt', linewidth=0.5)
    ax33.fill_betweenx(plotdata.TVDSS, plotdata.VCL, (1 - plotdata.PHIE), color='yellow', 
                            capstyle='butt', linewidth=0.5)
    ax33.fill_betweenx(plotdata.TVDSS, (1 - plotdata.PHIE), 1, color='gray', 
                            capstyle='butt', linewidth=0.5)
    ax33.set_xticks([0, 1])
    ax33.set_xticklabels(['', ''])

    # sand-shale lithology plot

    plotdata['liplot'] = np.nan
    plotdata.loc[plotdata.LITHO == 'SAND', 'liplot'] = 1
    plotdata.loc[plotdata.LITHO == 'SHALE', 'liplot'] = 0
        
    ax41 = axis4.twiny()
    ax41.fill_betweenx(plotdata.TVDSS, plotdata.liplot, 1, color='SaddleBrown', 
                            capstyle='butt', linewidth=0.01, label='SHALE')
    ax41.fill_betweenx(plotdata.TVDSS, 0, plotdata.liplot, color='yellow', 
                            capstyle='butt', linewidth=0.01, label='SAND')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('gray')
    ax41.set_xlim(0, 1)
    ax41.set_xlabel('LITHOLOGY', color = 'gray')
    ax41.tick_params(axis = 'x', colors = 'gray')
    ax41.set_xticks([0, 1])
    ax41.set_xticklabels(['', ''])
    ax41.legend(loc = 'upper left')

    plotdata.drop(columns = ['liplot'], inplace = True)

    fig.tight_layout()

    # Save files

    filename = '%s_ND.jpg' %wellname
    ndfolder = 'MecStrati'
    ndpath = os.path.join(savepath, ndfolder)

    if not os.path.isdir(ndpath):
        os.makedirs(ndpath)

    plt.savefig(os.path.join(ndpath, filename), dpi=300, format='jpg', bbox_inches="tight")
    plt.close(fig)

# Function for all overburden stress plot

def allobp_plot(**kwargs):
    """
    input:  dataframes = list of well logging data in dataframe,
            wellids = dictionary of name and color code,
            savepath = path to save folder
    """

    dataframes = kwargs.get('dataframes')
    wellids = kwargs.get('wellids')
    savepath = kwargs.get('savepath')

    import os
    import matplotlib.pyplot as plt

    # Create figure and subplots
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 10), sharey = True)
    fig.suptitle('Overburden Pressure', fontsize = 16, y = 1.0)
    
    # General setting for all axis

    maxscale = max([dataframe.OBP.max() for dataframe in dataframes])
    scale, scalelist = plotscale(maxscale=maxscale, minscale=0, increment=2000)

    # overburden pressure
    
    ax.set_ylabel('TVD[m]')
    ax.set_ylim(0, max([dataframe.TVD.max() for dataframe in dataframes]))
    ax.invert_yaxis()
    ax.minorticks_on() #Scale axis
    ax.get_xaxis().set_visible(False) 
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')

    ax.set_xlim(scale[0], scale[-1]) 
    for dataframe, name in zip(dataframes, wellids):
        ax.plot(dataframe.OBP, dataframe.TVD, color=wellids[name], linewidth='1', label=name)
    ax.set_xlabel('OBP[psi]', color='black')
    ax.tick_params(axis='x', colors='black')

    ax.set_xticks(scale)
    ax.set_xticklabels(scalelist)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid(True)
    ax.legend(loc = 'upper right')
        
    fig.tight_layout()

    # Save files

    filesave = 'All_OBP.jpg'
    obpfolder = 'OBP'
    obppath = os.path.join(savepath, obpfolder)

    if not os.path.isdir(obppath):
        os.makedirs(obppath)

    plt.savefig(os.path.join(obppath, filesave), dpi=300, format='jpg', bbox_inches="tight")
    plt.close(fig)



# # Function for initial plot for first inspection

# def inspection(dataframe, las):
#     """
#     inputs: dataframe = well logging in dataframe
#             las = well logging in las file
#     """

#     import matplotlib.pyplot as plt
#     import numpy as np

#     # create figure

#     fig, axis = plt.subplots(nrows = 1, ncols = len(dataframe.columns), figsize = (30,20), sharey = True)
    
#     units = [curve.unit for curve in las.curves]
#     index_unit = units.pop(0)

#     # plot setting for all axis

#     bottom_depth = dataframe.index.max()
#     top_depth = dataframe.index.min()

#     axis[0].set_ylabel('MD[%s]' %index_unit, fontsize = 15)

#     for ax, col, unit in zip(axis, dataframe.columns, units):
#         ax.set_ylim(top_depth, bottom_depth)
#         ax.invert_yaxis()
#         ax.minorticks_on() #Scale axis
#         ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
#         ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
#         ax.set_xlabel(col + '\n[%s]' %unit, fontsize = 15)

#         if (col == 'RT') or (col == 'MSFL'):
#             ax.plot(dataframe[col], dataframe.index, linewidth = '0.5')
#             ax.set_xscale('log')

#         elif col == 'BHF':

#             dataframe['bhf'] = np.nan
#             dataframe.loc[dataframe.BHF == 'BAD', 'bhf'] = 1
#             ax.fill_betweenx(dataframe.index, 0, dataframe.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
#             dataframe.drop(columns = ['bhf'], inplace = True)

#         elif col in ['TVD', 'TVDSS', 'AZI', 'ANG', 'CAL', 'BS']:
#             ax.plot(dataframe[col], dataframe.index, linewidth = '1.0')
        
#         else:
#             ax.plot(dataframe[col], dataframe.index, linewidth = '0.5')

#     fig.tight_layout()

#     plt.show()


if __name__ == "__main__":
    
    import lasio
    import numpy as np
    import pandas as pd

    path_las = '/Users/phumikrai/Documents/All Projects/1D-MEM/Saved files/LASfiles/A-1_py.las'
    path_top = '/Users/phumikrai/Documents/All Projects/1D-MEM/Sirikit field/Formation tops/A-1_top.csv'
    path_pres = '/Users/phumikrai/Documents/All Projects/1D-MEM/Sirikit field/Pressures/A-1_pp.csv'
    path_core = '/Users/phumikrai/Documents/All Projects/1D-MEM/Sirikit field/Cores/A-1_core.csv'
    
    las = lasio.read(path_las)
    las_df = las.df()
    las_df.replace(-9999.25, np.nan, inplace=True)
    datarange = (las.well['STRT'].value, las.well['STOP'].value, las.well['STEP'].value)

    top = pd.read_csv(path_top)
    pres = pd.read_csv(path_pres)
    core = pd.read_csv(path_core)
    


