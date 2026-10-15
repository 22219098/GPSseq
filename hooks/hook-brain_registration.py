from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = []
hiddenimports += collect_submodules("brain_registration")
hiddenimports += collect_submodules("voxelmorph")
hiddenimports += collect_submodules("neurite")

datas = []
datas += collect_data_files("brain_registration")