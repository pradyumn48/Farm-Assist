from tkinter import *
import tkinter
from PIL import ImageTk,Image
from Home import CropHome

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()
    root.resizable(0, 0)
    top = Crop(root)
    root.mainloop() #interface intilization

class Crop:
    def __init__(self, top=None):

            _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
            _fgcolor = '#000000'  # X11 color: 'black'
            _compcolor = '#d9d9d9'  # X11 color: 'gray85'
            _ana1color = '#d9d9d9'  # X11 color: 'gray85'
            _ana2color = '#d9d9d9'  # X11 color: 'gray85'
            font10 = "-family {Segoe UI Black} -size 10 -weight bold " \
                     "-slant roman -underline 0 -overstrike 0"
            font9 = "-family {Segoe UI Black} -size 12 -weight bold -slant" \
                    " roman -underline 0 -overstrike 0"

            window_height = 394
            window_width = 394
            screen_width = top.winfo_screenwidth()
            screen_height = top.winfo_screenheight()
            x_cordinate = int((screen_width / 2) - (window_width / 2))
            y_cordinate = int((screen_height / 2) - (window_height / 2))
            top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

            top.title("Crop Disease Prediction")
            top.configure(background="#ffffff")
            top.configure(highlightbackground="#d9d9d9")
            top.configure(highlightcolor="black")

            self.menubar = Menu(top, font="TkMenuFont", bg=_bgcolor, fg=_fgcolor)
            top.configure(menu=self.menubar)

            img = Image.open("images/bgcrop.jpg")
            self._img1 = ImageTk.PhotoImage(img)
            background = tkinter.Label(top, image=self._img1, bd=0)
            background.pack(fill='both')
            background.image = self._img1

            self.Label1 = Label(top)
            self.Label1.place(relx=0.23, rely=0.1, height=207, width=207)
            self.Label1.configure(activebackground="#f9f9f9")
            self.Label1.configure(activeforeground="black")
            self.Label1.configure(background="#ffffff")
            self.Label1.configure(disabledforeground="#a3a3a3")
            self.Label1.configure(foreground="#000000")
            self.Label1.configure(highlightbackground="#d9d9d9")
            self.Label1.configure(highlightcolor="black")
            img=Image.open("images/logocrop.jpg")
            image = img.resize((207, 207), Image.ANTIALIAS)
            self._img1 = ImageTk.PhotoImage(image)
            self.Label1.configure(image=self._img1)
            self.Label1.configure(text='''Label''')
            self.Label1.configure(width=207)

            self.Label2 = Label(top)
            self.Label2.place(relx=0.08, rely=0.61, height=31, width=347)
            self.Label2.configure(activebackground="#f9f9f9")
            self.Label2.configure(activeforeground="black")
            self.Label2.configure(background="#ffffff")
            self.Label2.configure(disabledforeground="#a3a3a3")
            self.Label2.configure(font=font9)
            self.Label2.configure(foreground="#000080")
            self.Label2.configure(highlightbackground="#d9d9d9")
            self.Label2.configure(highlightcolor="black")
            self.Label2.configure(text='''Crop Disease Predition''')

            self.Button1 = Button(top)
            self.Button1.place(relx=0.53, rely=0.79, height=42, width=138)
            self.Button1.configure(activebackground="#d9d9d9")
            self.Button1.configure(activeforeground="#000000")
            self.Button1.configure(background="#408080")
            self.Button1.configure(disabledforeground="#a3a3a3")
            self.Button1.configure(font=font10)
            self.Button1.configure(foreground="#ffffff")
            self.Button1.configure(highlightbackground="#d9d9d9")
            self.Button1.configure(highlightcolor="black")
            self.Button1.configure(pady="0")
            self.Button1.configure(text='''PROCEED''')
            self.Button1.configure(command=self.openhome)

    def openhome(self):
        global root
        self.Label1.destroy()
        self.Label2.destroy()
        self.Button1.destroy()
        top = CropHome(root)


if __name__ == '__main__':
    vp_start_gui()