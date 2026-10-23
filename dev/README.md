# dev/

Work in progress. **Not part of the installed package** and not shipped to PyPI.

- `workbench.py` — tkinter GUI for assembling spacecraft models. Under development.
- `geometryBuilder.py` — model-assembly helpers; used only by `workbench.py`.

Both still read paths with hardcoded Windows separators (`"\\"`), so they do not
work on macOS or Linux as written. That is one reason they are not distributed.

To run them, have `satlight` importable and run from the repository root:

    python dev/workbench.py
