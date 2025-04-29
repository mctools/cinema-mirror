#!/usr/bin/env python3
"""
A python-based 3rd-party dependencies management for Prompt, intended to be cross-platform.
"""

import os
import subprocess
from pathlib import Path
import wget
import shutil
import platform
import contextlib

git_domain = "https://code.ihep.ac.cn/cinema-developers/"
kds_url = "kdsource/-/archive/v0.1.0_prompt/kdsource-v0.1.0_prompt.tar.gz"
gidiplus_url = "gidiplus/-/archive/v3.28.22_prompt/gidiplus-v3.28.22_prompt.tar.gz"
vecgeom_git = "https://code.ihep.ac.cn/caixx/VecGeom.git"
vecgeom_branch = "cinema"
cpu_count = os.cpu_count()

src_root = Path(__file__).parent.parent.parent.parent
src_build_path = src_root/"build"
externalpath = src_root/"external"
external_src_parent = externalpath/"src"
external_build_path = externalpath/"build"
install_root = externalpath/"install"

def init_external_dir():
    if platform.system() == 'Linux':
        print("Running in", platform.system())
        result_git = subprocess.run(["rm","-rf", "build", "external"], cwd=str(src_root),
                                        capture_output=True, text=True)

def create_folder(path : Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {str(path)}")
    except PermissionError:
        print(f"Permission error when create '{str(path)}'")
    except Exception as e:
        print(f"Not known error: {e}")
    print("")
    return path

def download_from_url(git_domain, url, root: Path):
    result = wget.download(git_domain+url, out=str(externalpath))
    print("")
    print(git_domain+url)
    print("to ")
    print(result)
    shutil.unpack_archive(result, str(root))

    extract_upper = Path(result).stem
    extract_upper = Path(extract_upper).stem

    print(str(extract_upper))
    srcpath = root/extract_upper
    return srcpath

def download(name, url=None, git_url=None, git_branch="main"):
    create_folder(external_src_parent)

    if git_url:
        print("Using git repo: ", git_url)
        try:
            result_git = subprocess.run(["git", "clone", "-b", git_branch, git_url, name], cwd=str(external_src_parent),
                                        capture_output=True,  text=True)
        finally:
            print(f"Error captured: {result_git.stderr}")
        src_path = external_src_parent/name
    else:
        print("Using url (released pkg): ", url)
        src_path = download_from_url(git_domain, url, external_src_parent)
    return src_path

def install(name, cmakeargs=None, makeargs=None, url=None, git_url=None, git_branch="main"):

    dep_root = download(name, url, git_url, git_branch)

    buildpath = create_folder(external_build_path/name)
    try:
        if not makeargs:
            print("project uses cmake. ")
            result = subprocess.run(["cmake"] + cmakeargs + [str(dep_root)], cwd=str(buildpath) ,
                                        check=True, capture_output=True,  text=True)

            result = subprocess.run(["make", f"-j{cpu_count}"], cwd=str(buildpath) ,
                                        check=True, capture_output=True,  text=True)
            result = subprocess.run(["make", "install"], cwd=str(buildpath) ,
                                        check=True, capture_output=True,  text=True)
        elif makeargs and cmakeargs:
            raise TypeError("Only one of cmakeargs and makeargs should be provided.")
        elif makeargs and not cmakeargs:
            print("project uses make. ", "cwd=", str(dep_root))
            # print(["make"]+makeargs, sep=',')
            result = subprocess.run(["make"]+makeargs, cwd=str(dep_root) , capture_output=True,  text=True)
        else:
            raise NotImplementedError("Not Implemented.")
    except Exception as e:
        print(f"Error messages: ")
        print(e.stderr)

def prepare_dependencies():
    vecgeom_cmakeargs = [f"-DCMAKE_INSTALL_PREFIX={install_root}" ,"-DVECGEOM_BUILTIN_VECCORE=ON" ,
        "-DVECGEOM_FAST_MATH=OFF" , "-DBUILD_TESTING=OFF" ,"-DVECGEOM_GDML=ON" ,
        "-DVECGEOM_USE_NAVINDEX=ON"]
    install("vecgeom", cmakeargs=vecgeom_cmakeargs, git_url=vecgeom_git, git_branch=vecgeom_branch)

    kds_cmakeargs = [f"-DCMAKE_INSTALL_PREFIX={install_root}", "-DCMAKE_POLICY_VERSION_MINIMUM=3.24" ]
    install("KDSource", cmakeargs=kds_cmakeargs, url=kds_url)

    gidiplus_makeargs = ["-s", "install4prompt", f"-j{cpu_count//2}", 'SHELL=bash', 'CXXFLAGS=-std=c++11 -fPIC', 'CFLAGS=-fPIC', # fixme: dynamic SHELL
                         f"INSTALL_PREFIX={install_root}"]
    install("gidiplus", makeargs=gidiplus_makeargs, url=gidiplus_url)

def main_build():
    create_folder(src_build_path)
    lib_path = install_root/"lib"
    veccore_dir = install_root/"lib"/"cmake"/"VecCore"
    cinema_cmakeargs = [f"-DCMAKE_PREFIX_PATH={install_root}", f"-DVecCore_DIR={veccore_dir}", f"-DKDS_LIB={lib_path}",
                        f"-DGIDIPLUS_LIB={lib_path}", f"-DGIDIPLUS_INCDIR={lib_path}", ]
    result = subprocess.run(["cmake"]+cinema_cmakeargs+[".."], cwd=str(src_build_path), capture_output=True,  text=True)
    print(result.stdout)
    print(result.stderr)
    result = subprocess.run(["make", f"-j{cpu_count}"], cwd=str(src_build_path), capture_output=True,  text=True)
    print(result.stdout)
    print(result.stderr)

    sharedlib_from = src_build_path/"src"/"cxx"/"libprompt_core.so"
    sharedlib_from = sharedlib_from.resolve()
    sharedlib_to = Path("src/python/Cinema")
    sharedlib_to = sharedlib_to.resolve()

    shutil.copy2(str(sharedlib_from), str(sharedlib_to))


def main():
    print("Show nc")
    subprocess.run(["ncrystal-config", "--show", "cmakedir"])
    subprocess.run(["mcpl-config", "--show", "cmakedir"])
    init_external_dir()
    prepare_dependencies()
    main_build()

    
if __name__ == "__main__":
    main()