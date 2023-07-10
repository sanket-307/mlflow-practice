## Objective

Use the famous code refactoring tool to make python code standard and less worry about code formatting and more focus on core logic.

Three tools we are going to use in this assignment to do code refactor

### 1. isort
	isort your imports, so you don't have to.

### 2. Black
	Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, 	determinism, and freedom from pycodestyle nagging about formatting. You will save time and mental energy for more important matters.

### 3. flake8
	A python tool that glues together pep8, pyflakes, mccabe, and third-party plugins to check the style and quality of some python code


#### To install isort
**pip install isort**

#### To install black
**pip install black**

#### To install flake8
**pip install flake8**


## Important Note:

## As some coding standards are overlapping with other tools to avoid that create three files in root directory.

**1. pyproject.toml**
> [tool.isort]
> profile = "black"

**2. .isort.cfg**
> [setting]
> multi_line_output = 3
> include_trailing_comma = True
> force_grid_wrap = 0
> use_parentheses = True
> ensure_newline_before_comments = True
> line_length = 88

**3. tox.ini**
> [flake8]
> max-line-length = 88
> extend-ignore = E203


#### Command to run isort:
**isort nonstandardcode.py**

#### Command to run black:
**black nonstandardcode.py**

#### Command to run flake8:
**flake8 nonstandardcode.py**


##### Output of flake8 as requirement of assignment

(mle-dev) vagrant@vagrant-ubuntu-trusty-64:~/git_workspace/mle-training$ flake8 nonstandardcode.py
nonstandardcode.py:4:1: F401 'matplotlib as mpl' imported but unused
nonstandardcode.py:5:1: F401 'matplotlib.pyplot as plt' imported but unused
nonstandardcode.py:24:1: F811 redefinition of unused 'pd' from line 7
nonstandardcode.py:24:1: E402 module level import not at top of file
nonstandardcode.py:34:1: E402 module level import not at top of file
nonstandardcode.py:44:1: E402 module level import not at top of file
nonstandardcode.py:90:1: E402 module level import not at top of file
nonstandardcode.py:111:1: E402 module level import not at top of file
nonstandardcode.py:116:1: E402 module level import not at top of file
nonstandardcode.py:124:1: E402 module level import not at top of file
nonstandardcode.py:130:1: E402 module level import not at top of file
nonstandardcode.py:141:1: E402 module level import not at top of file
nonstandardcode.py:142:1: E402 module level import not at top of file
nonstandardcode.py:143:1: E402 module level import not at top of file
nonstandardcode.py:165:1: E402 module level import not at top of file
(mle-dev) vagrant@vagrant-ubuntu-trusty-64:~/git_workspace/mle-training$ flake8 nonstandardcode.py
(mle-dev) vagrant@vagrant-ubuntu-trusty-64:~/git_workspace/mle-training$ vim nonstandardcode.py



