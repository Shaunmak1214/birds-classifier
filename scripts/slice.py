import glob
import os
import argparse
import shutil

foldername = './dataset/test';

count = 0;

for subdir, dirs, files in os.walk(foldername):
  if (count > 0):
    print (subdir)
    shutil.copytree(subdir, './dataset/20test/' + subdir.split('/')[-1])
  
  count = count + 1;
  if(count >= 21):
    break;