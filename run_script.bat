@echo off

rem Activate the specific conda environment

call "C:\Users\nickt\anaconda3\Scripts\activate.bat" "C:\Users\nickt\anaconda3\envs\my-quant-stack"

rem Run test.py located in the same folder as this .bat, pass all args

python "%~dp0test.py" %*

rem Optional: deactivate when done

rem call conda deactivate