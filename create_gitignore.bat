@echo off
echo Creating .gitignore for Python projects...
echo # Python .gitignore > .gitignore
echo __pycache__/>> .gitignore
echo *.py[cod]>> .gitignore
echo *$py.class>> .gitignore
echo .ipynb_checkpoints/>> .gitignore
echo .vscode/>> .gitignore
echo venv/>> .gitignore
echo env/>> .gitignore
echo .venv/>> .gitignore
echo ENV/>> .gitignore
echo .DS_Store>> .gitignore
echo Thumbs.db>> .gitignore
echo *.~ipynb_checkpoints>> .gitignore
echo Done!
