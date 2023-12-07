import glob 
from setuptools import setup, find_packages
import os
import subprocess
from setuptools import Extension, setup
from distutils.command.build import build as _build
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):

    def run(self):     
        for ext in self.extensions:
            self.build_cmake_extension(ext)
        extensions = ['libprompt_core']
        self.copy_precompiled_files(extensions)

    def build_cmake_extension(self, ext):
        temp = vars(ext)
        for item in temp:
            print(item, ':', temp[item])

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
            
        build_temp = os.path.abspath(build_temp)
        print(f'build_temp: {build_temp}')

        numcpu = os.cpu_count()//2
                
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        if not os.path.exists(os.path.join(build_temp, 'install')):
            subprocess.check_call(['mkdir', 'install'] , cwd=build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] , cwd=build_temp+'/install')
        subprocess.check_call(['make', '-j'+str(numcpu)] , cwd=build_temp+'/install')
                
    def copy_precompiled_files(self, file_list, orig = None):
        
        import os
        import shutil
        from setuptools import find_packages

        if not orig:
            orig = self.build_temp

        pkg = find_packages(where = os.path.join("src", "python"))[0]

        for dir, subdirs, filenames in os.walk(os.path.join(orig)):
            for files in filenames:
                if files.split('.')[0] in file_list:
                    dest = os.path.join(os.path.abspath('.'), self.build_lib, pkg, files)
                    shutil.copyfile(os.path.join(dir, files), dest)

class Recorder(_build):
    def run(self):
        self.record_commit_hash()
        super().run()

    def record_commit_hash(self):
        import subprocess
        hash_value = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        with open('hash.txt', 'w') as f:
            f.write(hash_value)

def package_files(directory):

    dirs = []
    datalist = []
    for (path, directories, filenames) in os.walk(directory):
        datalist.append((os.path.join('/Cinema', path), 
                         [os.path.join(path, f) for f in filenames]))
        print((os.path.join('/Cinema', path), filenames))
    return datalist

datalist = package_files('gdml')

setup(
    
    packages=['Cinema', 'Cinema.Interface', 'Cinema.Prompt', 'Cinema.Prompt.Histogram'],

    package_dir={
        'Cinema': os.path.join('.','src','python') + os.path.sep + 'Cinema',
        'Cinema.Experiment': os.path.join('.','src','python') + os.path.sep + os.path.join('Cinema', 'Experiment', 'Analyser'),
        'Cinema.Experiment.Analyser': os.path.join('.','src','python') + os.path.sep + os.path.join('Cinema', 'Experiment'),
        'Cinema.Interface': os.path.join('.','src','python') + os.path.sep + os.path.join('Cinema', 'Interface'),
        'Cinema.Prompt': os.path.join('.','src','python') + os.path.sep + os.path.join('Cinema', 'Prompt'),
        'Cinema.Prompt.Histogram': os.path.join('.','src','python') + os.path.sep + os.path.join('Cinema', 'Prompt', 'Histogram'),
        },
    
    scripts=[os.path.join('.', 'scripts', 'prompt') ],

    data_files=[
        ('/Cinema/ncmat', glob.glob('./ncmat/*')),
        ('/Cinema/ncmat', glob.glob('./data_ncrystal/data/*')),
        ('/Cinema', glob.glob('./hash*')),
        ] + datalist,

    install_requires=['pyvista', 'matplotlib', 'mcpl'],

    ext_modules=[CMakeExtension(name='cinema', sourcedir='')],

    cmdclass={
        'build_ext': CMakeBuild,
        'build': Recorder,
        },
)
