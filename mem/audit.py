# function for decision confirmation

def answer():
    """
    output: either only 'Yes' or 'No' from user
    """
    
    while True:
        answer = input('Are you sure? [Yes/No]: ').strip()

        if answer.lower() == 'yes':
            break
        elif answer.lower() == 'no':
            break
        else:
            print('Please confirm again!')

    return answer

# function for getting file path from user input definition

def defpath(**kwargs):
    """
    input:  base = path base,
            file = file,
            filetype = file type
    ...
    output: file path from user
    """

    base = kwargs.get('base')
    file = kwargs.get('file')
    filetype = kwargs.get('filetype')

    import os.path as op

    while True:
        print('Which one is your %s directory?' %file)
        folder = input('Please indicate your %s directory name%s: ' %(file, filetype)).strip()

        if folder == '':
            print('Please type the directory name!')
        else:
            folderpath = op.join(base, folder)
            if op.isdir(folderpath):
                break
            else:
                print('Please try again, your directory \'%s\' is not found!' %folder)
    
    return folderpath

# function for well name extraction from file path

def filename(filelist):
    """
    input:  filelist = list of file paths
    ...
    output: list of file names
    """
    
    import os.path as op

    return [op.basename(file).split('_', 1)[0].lower() for file in filelist]

# function for grouping the files by well name of prefix

def grouping(**kwargs):
    """
    input:  las = list of well logging file path,
            dev = list of deviation file path,
            top = list of formation top file path,
            pres = list of pressure file path,
            core = list of core file path,
            drill = list of drilling test file path,
            mud = list of mud weight log file path
    ...
    output: grouped data by well name in dictionary
    """
    las = kwargs.get('las')
    dev = kwargs.get('dev')
    top = kwargs.get('top')
    pres = kwargs.get('pres')
    core = kwargs.get('core')
    drill = kwargs.get('drill')
    mud = kwargs.get('mud')

    wellname = filename(las)

    groups = {}
    
    for name in wellname:
        A = (name in filename(las))
        B = (name in filename(dev))
        C = (name in filename(top))
        D = (name in filename(pres))
        E = (name in filename(core))
        F = (name in filename(drill))
        G = (name in filename(mud))

        if A and B and C and D and E and F and G:
            groups[name] = {'las':las[filename(las).index(name)], 
                            'dev':dev[filename(dev).index(name)],
                            'top':top[filename(top).index(name)],
                            'pres':pres[filename(pres).index(name)],
                            'core':core[filename(core).index(name)],
                            'drill':drill[filename(drill).index(name)],
                            'mud':mud[filename(mud).index(name)]}

    return groups

# fucnction for random bright color list (color map)

def default_colors(n_color):
    """
    input:  n_color = number of color want to be generated
    ...
    output: list of color code (hex code) following 25 defaults.
    """

    import random

    defaults = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                '#808080', '#FF0000', '#00FFFF', '#800000', '#008080',
                '#FFFF00', '#FFBF00', '#0000FF', '#808000', '#000080',
                '#00FF00', '#FF00FF', '#008000', '#800080', '#C0C0C0']

    if int(n_color) > 25:
        colors = defaults.copy()

        while len(colors) != int(n_color):
            scrap_code = [''.join([random.choice('0123456789ABCDEF') for i in range(2)]), '00', 'FF']
            random.shuffle(scrap_code)
            color = '#' + ''.join(scrap_code)
            if color not in colors:
                colors.append(color)
    else:
        colors = defaults[0:int(n_color)]
    
    return colors