# importing the library
import os
# giving directory name
dirname = './ENDF-VIII.gnds-2.0/ENDF-VIII.0/neutrons/'

fn = []
# giving file extension
ext = ('.gnds.xml')
# iterating over all files
for files in os.listdir(dirname):
    if files.endswith(ext):
        fn.append(files) # printing file name of desired extension




import subprocess

fn.sort()
for f in fn:
    print(f)
    # if f=='n-008_O_016.endf.gnds.xml':
    #     continue  

    subprocess.run(f'processProtare.py -t 2.586e-8 -mc --hybrid {dirname+f} {"./prompt_data/"+f}', shell=True, check=False)

subprocess.run(f'buildMapFile.py -o neutron.map -l prompt2.0 prompt_data/*xml', shell=True, check=False)

