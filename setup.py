from setuptools import setup, find_packages

# https://github.com/pybind/cmake_example/blob/master/setup.py

import os
import shutil
import re
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', git=None, build_args_extra=''):
        Extension.__init__(self, name, sources=[])
        self.build_args_extra = build_args_extra
        if git:
            self.git = git
        else:
            self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        temp = vars(ext)
        for item in temp:
            print(item, ':', temp[item])

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
            

        build_temp = os.path.abspath(build_temp)
        print(f'build_temp {build_temp}')

        numcpu = os.cpu_count()//2

        try:

            if hasattr(ext,'git'):
                subprocess.check_call(['git', 'clone', ext.git, 'src'] , cwd=build_temp)
                subprocess.check_call(['mkdir', 'build'], cwd=build_temp)
                subprocess.check_call(['cmake', '-DCMAKE_INSTALL_PREFIX='+build_temp+'/install' , ext.build_args_extra, '../src'] , cwd=build_temp+'/build')
                subprocess.check_call(['make', '-j'+str(numcpu)], cwd=build_temp+'/build' )
                subprocess.check_call(['make', 'install'] , cwd=build_temp+'/build')
                os.environ.setdefault(ext.name, build_temp+'/install')
                print(f'{ext.name} is install into {build_temp+"/install"}')

            else:
                extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

                # required for auto-detection & inclusion of auxiliary "native" libs
                if not extdir.endswith(os.path.sep):
                    extdir += os.path.sep

                subprocess.check_call(['mkdir', 'install'] , cwd=build_temp)
                subprocess.check_call(["cmake", ext.sourcedir] , cwd=build_temp+'/install')
                subprocess.check_call(['make', '-j'+str(numcpu)] , cwd=build_temp+'/install')
                
        except:
            shutil.rmtree(build_temp)


#template from https://zhuanlan.zhihu.com/p/276461821
setup(
    python_requires='>=3.8',

    classifiers = [
        # 发展时期,常见的如下
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # 开发的目标用户
        'Intended Audience :: Developers',

        # 属于什么类型
        'Topic :: Software Development :: Build Tools',

        # 许可证信息
        'License :: OSI Approved :: MIT License',

        # 目标 Python 版本
        'Programming Language :: Python :: 3.8',
    ],
    name="Ncinema",
    version="0.1",
    author="x.x. cai",
    author_email="wongbingming@163.com",
    description="China Spallation Neutron Source Monte Carlo System",
    url="",
    
    packages=find_packages(
        where = os.path.join("src", "python")),

    package_dir=dict(
        zip(
            find_packages(where = os.path.join("src", "python")), 
            [
                os.path.join('.','src','python') + os.path.sep + str(i_dir).replace('.', os.path.sep) \
                for i_dir in find_packages(where = os.path.join("src", "python"))
                
                ]
            )
        ),
    
   #
   #  # 安装过程中，需要安装的静态文件，如配置文件、service文件、图片等
   #  data_files=[
   #      ('', ['conf/*.conf']),
   #      ('/usr/lib/systemd/system/', ['bin/*.service']),
   #             ],
   #
    # 希望被打包的文件
    # package_data={
    #     './build/temp.linux-x86_64-cpython-38/NCrystal_ext':['*.so']
    #            },
   #  # 不打包某些文件
   #  exclude_package_data={
   #      'bandwidth_reporter':['*.txt']
   #             }
   #
   #     # 表明当前模块依赖哪些包，若环境中没有，则会从pypi中下载安装
   #  install_requires=['docutils>=0.3'],
   #
   #  # setup.py 本身要依赖的包，这通常是为一些setuptools的插件准备的配置
   #  # 这里列出的包，不会自动安装。
   #  setup_requires=['pbr'],
   #
   #  # 仅在测试时需要使用的依赖，在正常发布的代码中是没有用的。
   #  # 在执行python setup.py test时，可以自动安装这三个库，确保测试的正常运行。
   #  tests_require=[
   #      'pytest>=3.3.1',
   #      'pytest-cov>=2.5.1',
   #  ],
   #
   #  # 用于安装setup_requires或tests_require里的软件包
   #  # 这些信息会写入egg的 metadata 信息中
   #  dependency_links=[
   #      "http://example2.com/p/foobar-1.0.tar.gz",
   #  ],
   #
   #  # install_requires 在安装模块时会自动安装依赖包
   #  # 而 extras_require 不会，这里仅表示该模块会依赖这些包
   #  # 但是这些包通常不会使用到，只有当你深度使用模块时，才会用到，这里需要你手动安装
   #  extras_require={
   #      'PDF':  ["ReportLab>=1.2", "RXP"],
   #      'reST': ["docutils>=0.3"],
   #  }
   #
   # #  关于 install_requires， 有以下五种常用的表示方法：
   # # 1. 'argparse'，只包含包名。 这种形式只检查包的存在性，不检查版本。 方便，但不利于控制风险。
   # # 2. 'setuptools==38.2.4'，指定版本。 这种形式把风险降到了最低，确保了开发、测试与部署的版本一致，不会出现意外。 缺点是不利于更新，每次更新都需要改动代码。
   # # 3. 'docutils >= 0.3'，这是比较常用的形式。 当对某个库比较信任时，这种形式可以自动保持版本为最新。
   # # 4. 'Django >= 1.11, != 1.11.1, <= 2'，这是比较复杂的形式。 如这个例子，保证了Django的大版本在1.11和2之间，也即1.11.x；并且，排除了已知有问题的版本1.11.1（仅举例）。 对于一些大型、复杂的库，这种形式是最合适的。
   # # 5. 'requests[security, socks] >= 2.18.4'，这是包含了额外的可选依赖的形式。 正常安装requests会自动安装它的install_requires中指定的依赖，而不会安装security和socks这两组依赖。 这两组依赖是定义在它的extras_require中。 这种形式，用在深度使用某些库时。
   #
   #  # 用来支持自动生成脚本，安装后会自动生成 /usr/bin/foo 的可执行文件
   #  # 该文件入口指向 foo/main.py 的main 函数
   #  entry_points={
   #      'console_scripts': [
   #          'foo = foo.main:main'
   #      ]
   #  },
   #
   #  # 将 bin/foo.sh 和 bar.py 脚本，生成到系统 PATH中
   #  # 执行 python setup.py install 后
   #  # 会生成 如 /usr/bin/foo.sh 和 如 /usr/bin/bar.py
   #  scripts=['bin/foo.sh', 'bar.py']

    ext_modules=[CMakeExtension(name='NCrystal_ext', git='https://gitlab.com/xxcai1/ncrystal.git'),
                 CMakeExtension(name='Vecgeom_ext', git='https://gitlab.com/xxcai1/VecGeom.git', build_args_extra='-DGDML=On -DUSE_NAVINDEX=On'),
                 CMakeExtension(name='cinema', sourcedir='')],
    cmdclass={"build_ncrystal": CMakeBuild,
            "build_vecgeo": CMakeBuild,
            "build_cinema": CMakeBuild, },
)
