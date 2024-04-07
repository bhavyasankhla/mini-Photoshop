import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from collections import Counter
import struct
import utils 
import core_operations as core
from core_operations import convert_to_grayscale_manual


def parse_bmp(file_path):
    with open(file_path, 'rb') as bmp_file:
        # Read the BMP Header
        bmp_header = bmp_file.read(14)
        header_fields = struct.unpack('<2sIHHI', bmp_header)
        file_type, file_size, reserved1, reserved2, offset = header_fields
        if file_type != b'BM':
            raise ValueError('This is not a BMP file.')

        # Read the DIB Header
        dib_header = bmp_file.read(40)
        dib_fields = struct.unpack('<IiiHHIIIIII', dib_header)
        header_size, width, height, planes, bits_per_pixel, compression, image_size, x_ppm, y_ppm, colors, important_colors = dib_fields

        if bits_per_pixel != 24 or compression != 0:
            raise ValueError('Unsupported BMP format. Only 24-bit RGB full color uncompressed mode is supported.')

        # Ensure within specified dimensions
        if not (1 <= width <= 1024) or not (1 <= height <= 768):
            raise ValueError('Image dimensions are out of the allowed range.')

        print(f"Image Dimensions: {width}x{height}")
        
        # Move to the start of the pixel array
        bmp_file.seek(offset)

        # Calculate the padding at the end of each row
        row_padding = (4 - (width * 3) % 4) % 4

        # Read and store pixel data
        pixel_array = []
        for row in range(height):
            row_data = []
            for col in range(width):
                b, g, r = struct.unpack('BBB', bmp_file.read(3))
                row_data.append((r, g, b))
            # Skip the padding
            bmp_file.read(row_padding)
            pixel_array.insert(0, row_data)  # BMP files store pixel data bottom-up

        return pixel_array


def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if not filepath:
        return
    if filepath.lower().endswith('.bmp'):
        pixel_data = parse_bmp(filepath)
        height, width = len(pixel_data), len(pixel_data[0])
        data = [pixel for row in pixel_data for pixel in row]
        image = Image.new('RGB', (width, height))
        image.putdata(data)
        app.display_image(image)
    

def exit_program():
    root.destroy()


class PhotoshopApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, bg='grey')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        fileMenu = tk.Menu(menubar)
        coreMenu = tk.Menu(menubar)
        optionalMenu = tk.Menu(menubar)

        menubar.add_cascade(label="File", menu=fileMenu)
        menubar.add_cascade(label="Core Operations", menu=coreMenu)
        menubar.add_cascade(label="Optional Operations", menu=optionalMenu)

        fileMenu.add_command(label="Open File", command=open_file)
        fileMenu.add_command(label="Exit", command=exit_program)

        coreMenu.add_command(label="Grayscale", command=lambda: self.apply_effect(convert_to_grayscale_manual))
        coreMenu.add_command(label="Ordered Dithering", command=lambda: self.apply_effect_ordered_dithering(core.ordered_dithering))
        coreMenu.add_command(label="Auto Level", command=lambda: self.apply_effect(core.auto_level))
        coreMenu.add_command(label="Huffman", command=lambda: self.apply_huffman(core.show_huffman_info))

        optionalMenu.add_command(label="Flip Horizontal", command=lambda: self.apply_effect(utils.flip_horizontal))        
        optionalMenu.add_command(label="Flip Vertical", command=lambda: self.apply_effect(utils.flip_vertical))        
        optionalMenu.add_command(label="Solarize", command=lambda: self.apply_effect(utils.solarize))
        optionalMenu.add_command(label="Edge Detection", command=lambda: self.apply_effect(utils.edge_detection))        
        optionalMenu.add_command(label="Vignette", command=lambda: self.apply_effect(utils.vignette))        
        optionalMenu.add_command(label="Black & White", command=lambda: self.apply_effect(utils.thresholding))
        optionalMenu.add_command(label="Gaussian Blur", command=lambda: self.apply_effect(utils.gaussian_blur_manual))                
        optionalMenu.add_command(label="Warm Image", command=lambda: self.apply_effect(utils.color_balance_warm))        
        optionalMenu.add_command(label="Cool Image", command=lambda: self.apply_effect(utils.color_balance_cool))        
        optionalMenu.add_command(label="Reduce Noise", command=lambda: self.apply_effect(utils.noise_reduction))        
        

    def display_image(self, image):
        self.image = image
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def apply_effect(self, effect):
        if not hasattr(self, 'image'):
            messagebox.showerror("Error", "No image loaded")
            return

        original = self.image
        modified = effect(self.image)

        original_photo = ImageTk.PhotoImage(original)
        modified_photo = ImageTk.PhotoImage(modified)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
        self.canvas.create_image(original.width, 0, anchor=tk.NW, image=modified_photo)

        self.photo = (original_photo, modified_photo)


    def apply_huffman(self, effect):
        if not hasattr(self, 'image'):
            messagebox.showerror("Error", "No image loaded")
            return

        original = self.image
        effect(self.image)
        
    
    def apply_effect_ordered_dithering(self, effect):
        if not hasattr(self, 'image'):
            messagebox.showerror("Error", "No image loaded")
            return

        original = self.image
        grayscale = convert_to_grayscale_manual(original)
        modified = effect(original)

        original_photo = ImageTk.PhotoImage(grayscale)
        modified_photo = ImageTk.PhotoImage(modified)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
        self.canvas.create_image(grayscale.width, 0, anchor=tk.NW, image=modified_photo)

        # Update image references to prevent garbage collection
        self.photo = (original_photo, modified_photo)


root = tk.Tk()
root.title("Mini Photoshop")
root.geometry("1200x800")
app = PhotoshopApp(root)
root.mainloop()