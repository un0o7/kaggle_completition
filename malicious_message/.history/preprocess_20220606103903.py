# -*- encoding: utf-8 -*-
'''
@Author:  un0o7
@Date:  2022-06-06 10:36:10
@Last Modified by:  un0o7
@Last Modified time:  2022-06-06 10:36:10
'''

# replace chinese special characters with english
# space remove
#

import pandas as pd

train_df = pd.read_csv('./data/train_public.csv', seq='\t')
texts = train_df.iloc[:, 1].tolist()
result = train_df.iloc[:, 2].tolist()
