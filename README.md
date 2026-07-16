# St Kilda Project

## Prerequisites

* **Python:** Version `3.12.10` is recommended.

---

## Setup Instructions

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

Open your terminal and clone the repository to your device by running:

```bash
git clone https://github.com/biomonash/stkilda.git
cd stkilda

```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to isolate project dependencies. You can do this via the Command Line or directly within VS Code.

#### Option A: Using the Command Line

1. Create the virtual environment (named `venv`) by running:
```bash
python -m venv venv

```


2. Activate the virtual environment based on your operating system and terminal:
* **On Windows (Command Prompt / PowerShell):**
```cmd
venv\Scripts\activate

```


* **On Windows (Git Bash):**
```bash
source venv/Scripts/activate

```


* **On macOS and Linux:**
```bash
source venv/bin/activate

```


> 💡 **Tip:** If you are unsure which command to use, open the generated `venv` folder. If it contains a `Scripts` folder, use  `venv/Scripts/activate` as the path in the command. If it contains a `bin` folder, use `venv/bin/activate` as the path in the command.
> You will know it is running successfully when you see `(venv)` or `((venv))` at the beginning of your terminal prompt.



#### Option B: Using VS Code

If you are using Visual Studio Code, you can automate this process:

1. Open the Command Palette (`Ctrl+Shift+P` on Windows/Linux, `Cmd+Shift+P` on macOS).
2. Type and select **Python: Create Environment**.
3. Select **Venv**.
4. Choose your installed Python interpreter (e.g., Python 3.12.10).
5. Open a new terminal in VS Code (`Ctrl+Shift+``), and it will activate the virtual environment automatically.

### 3. Install Dependencies

Once your virtual environment is active, upgrade `pip` and install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

### 4. Testing Your Code

To test your scripts or experiment with the data:

1. Create a folder named `test` (or similar) at the root level of the project.
2. You can create standard Python files (`.py`) or Jupyter Notebooks (`.ipynb`) inside this folder to safely test your code without altering the main project files.
3. Most of the project folders contain classes and are implemented as Python modules (indicated by the presence of an `__init__.py` file). You can import these classes directly into your test files.
4. **Important:** Because these folders are structured as modules, if you want to execute a script located inside one of them, you should run it using the module flag (`-m`) from the root directory instead of running the file path directly.
For example, run:
```bash
python -m folder_name.file_name

```


Instead of:
```bash
python folder_name/file_name.py

```

*This ensures that all relative imports and classes within the module resolve correctly.*