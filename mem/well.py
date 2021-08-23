"""

borehole class containing well logging and support data

"""

from mem.alias import *

class borehole:
    def __init__(self, **kwargs):
        """
        input:  las = well logging data (.las),
                dev = deviation data (.csv),
                top = formation top data (.csv),
                pres = pressure data (.csv),
                core = core data (.csv),
                drill = drilling test data (.csv),
                mud = mud-weight data (.csv),
                color = identity color gererated by default_colors function from subfile.py
        ...
        output: well object containing input data for modeling
        """

        import lasio
        
        # imported borehole data

        self.las = kwargs.get('las')
        self.dev = kwargs.get('dev')
        self.top = kwargs.get('top')
        self.pres = kwargs.get('pres')
        self.core = kwargs.get('core')
        self.drill = kwargs.get('drill')
        self.mud = kwargs.get('mud')
        self.color = kwargs.get('color')
        
        # extracted data

        self.name = self.las.well['WELL'].value # well name
        self.df = self.las.df() # pandas dataframe
        self.range = (self.las.well['STRT'].value, self.las.well['STOP'].value, self.las.well['STEP'].value)

        # setup data

        self.df.rename_axis('MD', inplace=True)
        self.las.insert_curve(0, 'MD', self.df.index, unit = 'm', descr = 'Measured Depth', value = '')
        del self.las.curves['DEPTH']
        self.other = {} # store for any utility
        
        # user determination

        self.type = None # field type either onshore or offshore
        self.kb = None # kelly bushing
        self.gl = None # ground level
        self.wl = None # water level
        self.ml = None # mud line density
        self.ag = None # air gap

        # check completion
        
        datalist = [(self.las, self.df), self.dev, self.top, self.pres,
                    self.core, self.drill, self.mud]

        aliaslist = [curve_alias, dev_alias, top_alias, pres_alias,
                    core_alias, drill_alias, mud_alias]

        datatype = ['Well logging', 'Deviation', 'Formation top', 'Pressure test',
                    'Core test', 'Drilling test', 'Mud weight log']

        checked = []

        for data, dalias, dtype in zip(datalist, aliaslist, datatype):
            completion = setncheck(data=data, dalias=dalias, dtype=dtype, name=self.name)
            checked.append(completion)

        self.completion = all(checked)

    # function for exporting the data

    def export(self, **kwargs):
        """
        save_path = path to saved folder
        ...
        export new las file (.las) and comma-separated values file (.csv)
        """
        save_path = kwargs.get('save_path')

        import os, datetime
        import lasio

        # create a new empty las file and export to the new one

        las_file = lasio.LASFile()
        las_file.set_data(self.df)

        # update curve unit and its description

        for curve_1, curve_2 in zip(las_file.curves, self.las.curves):
            if curve_1.mnemonic == curve_2.mnemonic:
                curve_1.unit = curve_2.unit
                curve_1.descr = curve_2.descr

        # update header

        las_file.well = self.las.well

        # note in las file

        las_file.other = 'This file was written by python in %s' %datetime.date.today().strftime('%m-%d-%Y')

        # setup header for csv file

        headers = []

        for curve in self.las.curves:
            header = '%s[%s]' %(curve.mnemonic, curve.unit)
            headers.append(header)

        index = headers.pop(0)

        # export las and csv files

        las_folder, csv_folder = 'LASfiles', 'CSVfiles'

        for folder in [las_folder, csv_folder]:
            if not os.path.isdir(os.path.join(save_path, folder)):
                os.makedirs(os.path.join(save_path, folder))

        las_file.write(os.path.join(save_path, las_folder, 
                                    '%s_py.las' %self.name), version = 2.0) # export las
        
        self.df.rename_axis(index).to_csv(os.path.join(save_path, csv_folder, 
                                                        '%s_py.csv' %self.name), header = headers) # export csv

def setncheck(**kwargs):
    """
    input:  data = data for checking in dataframe
            dalias = alias for data
            dtype = data type
            name = well name
    ...
    output: the standard data with alias applied and check the completion
    """

    import random

    data = kwargs.get('data')
    dalias = kwargs.get('dalias')
    dtype = kwargs.get('dtype')
    name = kwargs.get('name')

    # setup new column name

    if isinstance(data, tuple):
        managelas = True
        setdata = data[1]
        lasdata = data[0]
    else:
        managelas = False
        setdata = data

    new_cols = {}
    seen = {}

    for col in setdata:
        for key, values in dalias.items():
            if col in values:
                new_cols[col] = key
                if key not in seen:
                    seen[key] = list([col])
                else:
                    seen[key].append(col)

    # manange duplicate

    for new_col in seen:
        n_data = len(seen[new_col])
        if n_data != 1:
            for col in seen[new_col]:
                new_cols.pop(col)
            old_col = random.choice(seen[new_col])
            new_cols[old_col] = new_col
            print('Randomly select %s from %d of them for well %s' %(new_col, n_data, name)) 
    
    # apply standard alias to dataframe

    setdata.rename(columns=new_cols, inplace=True)

    # eliminate unnessesary data

    unnessesary = []

    for col in setdata:
        if col not in dalias:
            setdata.drop([col], axis=1, inplace=True)
            unnessesary.append(col)

    # manage las file

    if managelas == True:
        for key, value in new_cols.items():
            lasdata.curves[key].mnemonic = value # apply standard alias
        for delcol in unnessesary:
            del lasdata.curves[delcol] # eliminate unnessesary data

    # check the completion

    if set(dalias).issubset(set(setdata)):
        completion = True
    else:
        completion = False
        print('%s data of well %s is not completed.' %(dtype, name))

    return completion
