import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Label 下划线示例")

label = ttk.Label(root, text="This is a _label with underline", underline=10)
label.pack(padx=10, pady=10)

root.mainloop()