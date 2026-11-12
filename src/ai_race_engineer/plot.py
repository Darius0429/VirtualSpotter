import pandas as pd
df = pd.read_fwf('/home/quang/projects/untitled.txt', header=None,
            colspecs=[[0,20], [22,34], [35,39], [42, 54], [54,-1]],   # modify this to fit your needs
            parse_dates=[0],
            names=['time', 'veloc', 'value', 'location', 'type']    # also modify this
           ) 