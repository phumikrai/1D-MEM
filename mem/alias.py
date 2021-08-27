"""

all alias for the variable from imported files

Note: only items in dictionary are allowed to be changed

"""

# well logging data

curve_alias = {
'BS':['BS', 'BIT'],
'CAL':['CAL', 'CALI', 'CALS', 'CLDC'],
'GR':['GR', 'GRGC', 'GAM'],
'RHOB':['RHOB', 'DEN', 'DENS'],
'NPHI':['NPHI', 'NPOR'],
'MSFL':['MSFL', 'R20T', 'RSHAL', 'RESS'],
'RT':['RT', 'R85T', 'LLD', 'RESD'],
'DTC':['DTC', 'DT35', 'DT'],
'DTS':['DTS', 'DTSM', 'DTSRM', 'DTSXX_COL', 'DTSYY_COL']
}

# deviation data

dev_alias = {
'MD':['MD', 'DEPTH'],
'AZI':['AZIMUTH', 'AZI'],
'ANG':['ANGLE', 'ANG']
}

# well configuration

config_alias = {
'TYPE':['TYPE', 'FIELD'],
'KB':['KB',],
'GL':['GROUND', 'GL'],
'WL':['WATER', 'WL'],
'ML':['Mudline', 'ML']
}

# formation top data

top_alias = {
'FORM':['FORMATION', 'FORMATIONS'],
'TOP':['TOP', 'DEPTH'],
'BOT':['BOTTOM', 'BOT']
}

# pressure data

pres_alias = {
'MD':['MD', 'DEPTH'],
'PP':['PRESSURE', 'PP']
}

# core data

core_alias = {
'MD':['MD', 'DEPTH'],
'YME':['YME',],
'UCS':['UCS',]
}

# drilling test data

drill_alias = {
'MD':['MD', 'DEPTH'],
'TYPE':['TEST', 'TYPE'],
'VAL':['RESULT', 'VALUE']
}

# mud weight data

mud_alias = {
'MD':['MD', 'DEPTH'],
'MW':['MUDWEIGHT', 'MW'],
'ECD':['ECD',]
}