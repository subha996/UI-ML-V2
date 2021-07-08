import numpy as np
import pandas as pd

## This fuunction will convert classification report in to Dataframe.

def classification_report_to_dataframe(str_representation_of_report):
    """A function wich convert classififcation report in to DataFrame"""
    try:
        split_string = [x.split(' ') for x in str_representation_of_report.split('\n')]
        column_names = ['']+[x for x in split_string[0] if x!='']
        values = []
        for table_row in split_string[1:-1]:
            table_row = [value for value in table_row if value!='']
            if table_row!=[]:
                values.append(table_row)
        for i in values:
            for j in range(len(i)):
                if i[1] == 'avg':
                    i[0:2] = [' '.join(i[0:2])]
                if len(i) == 3:
                    i.insert(1,np.nan)
                    i.insert(2, np.nan)
                else:
                    pass
        report_to_df = pd.DataFrame(data=values, columns=column_names)
        return report_to_df
    except Exception as ex:
        print(str(ex))