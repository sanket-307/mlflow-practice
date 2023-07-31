# mlhousing

this project is about median house value prediction. median house value prediction has been done using four algorithm namely linear regression, decision tree, random forest with best hyperparamertes from random search technique and random forest with best hyperparamertes from grid search technique.

to run project git clone mlhousing

install vertiual environement with all required libraries using command

```python
conda env create --name mle-dev --file=/mlhousing/deploy/conda/env.yml

```

run process.py with optional command line arguments.

```
python process.py
```

run process.py with command line arguments.

python process.py -i `<input folder path where you want to save donwloaded data> `-f `<housing.csv> -r <output folder path where you want to save your raw data> -o <output folder path where you want to save processed data> -p <input folder path of processed training data> -m <artifcats path where you want to save artifacts>`

arguments help:

* -i `<input folder path where you want to save donwloaded data> `
* -f `<housing.csv> `
* `-r <output folder path where you want to save your raw data> `
* `-o <output folder path where you want to save processed data> `
* `-p <input folder path of processed training data> `
* `-m <artifcats path where you want to save artifacts>`

---

if process.py file run without command line arguments default argument will be used to run mlhousing project on successfull run;

you can find

* processed and feature engineering traininig data and labels in *data/processed folder*
* processed and feature engineering testing data and labels in *data/processed folder*
* 4 saved model namely *linearmodel.pkl, dtmodel.pkl , rf_rs_model.pkl , rf_grid_model.pkl* in *artifacts* folder

***Note :***

***rf_rs_model.pkl , rf_grid_model.pkl*** is not uploaded in github version as it is large in size

---

**Unit Test, Integration test and Installation Test**

---

run below command which will do unit testing, integration testing, installation test and generate coverage report.

```
coverage run -m pytest -v && coverage report -m
```

*Note:*

***conftest.py*** *file contain fixtures function which can be used in different tests.*

conftest file configured to take optional command line arguments for unit_test and intergration test.

to run test files with custom arguments

pytest - v tests/unit_tests/test_data_processing.py --i `<raw dataset folder path>`--f `<raw csv file name>`  --o `<processed training and testing data folder path>`  --m `<artifact folder path>`

---

**Style testing**

---

Style testing has been performed using isort, black and flake8 the configuration for each style test are in *setup.cfg*
# VScode Workspace settings json #

## paste below settings to vscode/setting.json file


```json

    {
        "editor.formatOnSave": true,
        "python.formatting.provider": "none",
        "python.formatting.blackArgs": [
            "--line-length=120"
        ],
        "isort.args": [
            "--profile=black"
        ],
        "[python]": {
            "editor.codeActionsOnSave": {
                "source.organizeImports": true
            },
            "editor.defaultFormatter": "ms-python.black-formatter"
        },
        "flake8.args": [
            "--max-line-length=120",
            // "--ignore=E402,F841,F401,E302,E305"
        ],
        "files.trimTrailingWhitespace": true,
        // E402: Module level import not at top of file
        // F841: Local variable is assigned to but never used
        // F401: Module imported but unused
        // E302: Expected 2 blank lines, found 0
        // E305: Expected 2 blank lines after class or function definition, found 1
    }
