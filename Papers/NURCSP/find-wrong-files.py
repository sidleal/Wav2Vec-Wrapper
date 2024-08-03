import os
from os import listdir
from os.path import isfile, join
import pandas as pd

df = pd.read_csv('d:/datasets/NURC-SP/metadata.csv')

print(df.head())

dirs_level1 = [f for f in listdir('d:/datasets/NURC-SP') if not isfile(join('d:/datasets/NURC-SP', f))]
for dir in dirs_level1:
    print(dir)
    dirs_level2 = [f for f in listdir(f'd:/datasets/NURC-SP/{dir}') if not isfile(join(f'd:/datasets/NURC-SP/{dir}', f))]
    for dir2 in dirs_level2:
        print(dir2)
        onlyfiles = [f for f in listdir(f'd:/datasets/NURC-SP/{dir}/{dir2}') if isfile(join(f'd:/datasets/NURC-SP/{dir}/{dir2}', f))]
        for f in onlyfiles:
            if df.query(f'file_name == "{dir}/{dir2}/{f}"')['file_name'].count() < 1:
                print("--", f"{dir}/{dir2}/{f}")
                os.remove(f'd:/datasets/NURC-SP/{dir}/{dir2}/{f}')
