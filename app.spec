from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

hiddenimports = []
hiddenimports += collect_submodules("brain_registration")
hiddenimports += collect_submodules("voxelmorph")
hiddenimports += collect_submodules("neurite")
hiddenimports += collect_submodules("tensorflow")
hiddenimports += collect_submodules("keras")
hiddenimports += collect_submodules("SimpleITK")
hiddenimports += collect_submodules("nibabel")
hiddenimports += collect_submodules("skimage")
hiddenimports += collect_submodules("scipy")

datas = []
datas += collect_data_files("brain_registration")
datas += [
    ("assets/model.png", "assets"),
    ("assets/workflow.png", "assets"),
]

binaries = []
binaries += collect_dynamic_libs("tensorflow")
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("SimpleITK")

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["hooks"],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BrainImageTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="BrainImageTool",
)