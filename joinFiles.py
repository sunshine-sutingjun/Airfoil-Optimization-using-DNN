import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def python_files_to_notebook_with_headers(source_dir, notebook_name):
    """
    Combine Python files from a directory into a Jupyter notebook with headers.

    Parameters:
    - source_dir: Directory containing Python files.
    - notebook_name: Name of the resulting Jupyter notebook file.
    """
    # Create a new notebook object
    notebook = new_notebook()
    notebook.cells = []  # Initialize the list of cells

    # List all Python files in the source directory
    python_files = [f for f in os.listdir(source_dir) if f.endswith('.py')]
    python_files.sort()  # Sort the files if needed

    # Add each Python file's content as a code cell with a header to the notebook
    for file_name in python_files:
        header_text = f"## {file_name}\n\nThis code is from the file `{file_name}`."
        notebook.cells.append(new_markdown_cell(header_text))  # Add Markdown cell for the header

        file_path = os.path.join(source_dir, file_name)
        with open(file_path, 'r') as file:
            code = file.read()
            notebook.cells.append(new_code_cell(code))  # Add code cell for the file content

    # Write the notebook to a new .ipynb file
    nbformat.write(notebook, f"{notebook_name}.ipynb")

# Example usage
source_directory = './'
output_notebook_name = 'Airfoil-Optimization-using-DNN'
python_files_to_notebook_with_headers(source_directory, output_notebook_name)
