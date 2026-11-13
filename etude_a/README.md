## full study (10 seeds) + analysis

```bash
bash run_experiments.sh
```

**Windows**: the `pysupertuxkart2` wheels do not ship their DLLs. Put them in `stk_deps/dependencies-win-x86_64/bin`, they are loaded automatically by `train.py`
- unzip SuperTuxKart's official [dependencies-win-x86_64.zip](https://github.com/supertuxkart/dependencies/releases/download/preview/dependencies-win-x86_64.zip) into `stk_deps/`, and copy `libz.dll` to `libzlib.dll`
- add `libc++.dll` and `libunwind.dll` from `x86_64-w64-mingw32/bin` of an [llvm-mingw](https://github.com/mstorsjo/llvm-mingw/releases) *msvcrt-x86_64* release

STK must run in its own process (`use_subprocess=True`, the default): in the same process as CUDA it segfaults
pystk2 also crashes at random when a new race is created
`train.py` then restarts STK and creates the race again
