@echo off

rmdir /S /Q dist
pyinstaller soundspec.py
copy soundspec-batch.bat dist\soundspec\
