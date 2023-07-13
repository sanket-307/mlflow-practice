# VScode Workspace settings json #

## paste below settings to vscode/setting.json file


''' json

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

''' json
