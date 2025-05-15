# Импортируем необходимые библиотеки
import rasterio
import os
import numpy as np
import torch
from PIL import Image
import fastai
from fastai.vision.all import *
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib.colors
import tkinter as tk
from tkinter import filedialog, Label, ttk

# Разрешаем деление на 0 для расчета индексов
np.seterr(divide='ignore', invalid='ignore')

# Функция загрузки изображеня
def imageUploader():
    fileTypes = [('Image files', '*.tif')] #*.png;*.jpg;*.jpeg;
    global path
    path = filedialog.askopenfilename(filetypes=fileTypes)

# Функция расчета индексов
def calculate():
    # Создаем новое окно для отображения
    global last_culc
    index = combobox.get()
    window_culc = Tk()
    window_culc.geometry('800x600')
    
    # Если выбран NDVI
    if index == 'NDVI':
        window_culc.title('NDVI culculation') # название окна
        ndvi = NDVI_culc(path) # расчет индекса
        last_culc = ndvi # глобальная переменная для сохранения результата
        plot_index(ndvi, "NDVI") # построение графика matplotlib
        
    # Если выбран SAVI
    if index == 'SAVI':
        window_culc.title('SAVI culculation') # название окна
        savi = SAVI_culc(path) # расчет индекса
        last_culc = savi # глобальная переменная для сохранения результата
        plot_index(savi, "SAVI") # построение графика matplotlib
        
    # Если выбран MSAVI
    if index == 'MSAVI':
        window_culc.title('MSAVI culculation') # название окна
        msavi = MSAVI_culc(path) # расчет индекса
        last_culc = msavi # глобальная переменная для сохранения результата
        plot_index(msavi, "MSAVI") # построение графика matplotlib   
         
    # Если выбран GNDVI
    if index == 'GNDVI':
        window_culc.title('GNDVI culculation') # название окна
        gndvi = GNDVI_culc(path) # расчет индекса
        last_culc = gndvi # глобальная переменная для сохранения результата
        plot_index(gndvi, "GNDVI") # построение графика matplotlib   
        
    # Если выбран NDWI
    if index == 'NDWI':
        window_culc.title('NDWI culculation') # название окна
        ndwi = NDWI_culc(path) # расчет индекса
        last_culc = ndwi # глобальная переменная для сохранения результата
        plot_index(ndwi, "NDWI") # построение графика matplotlib 
        
    # Если выбран ARVI
    if index == 'ARVI':
        window_culc.title('ARVI culculation') # название окна
        arvi = ARVI_culc(path) # расчет индекса
        last_culc = arvi # глобальная переменная для сохранения результата
        plot_index(arvi, "ARVI") # построение графика matplotlib    
        
    # Если выбран RVI
    if index == 'RVI':
        window_culc.title('RVI culculation') # название окна
        rvi = RVI_culc(path) # расчет индекса
        last_culc = rvi # глобальная переменная для сохранения результата
        plot_index(rvi, "RVI", 0, 10) # построение графика matplotlib    
        
    # Если выбран EVI
    if index == 'EVI':
        window_culc.title('EVI culculation') # название окна
        evi = EVI_culc(path) # расчет индекса
        last_culc = evi # глобальная переменная для сохранения результата
        plot_index(evi, "EVI") # построение графика matplotlib    
        
    # Загрузка изображения для отображения
    image = PhotoImage(file="data/1.png", master=window_culc)
    image_label = tk.Label(window_culc, image=image)
    image_label.pack(side=tk.TOP, pady=10)
    
    # Кнопка помощи
    helpButton = tk.Button(window_culc, text='Help', activebackground="blue", activeforeground="white", command=help_ind)
    helpButton.pack(side=tk.RIGHT, padx=10, pady=10)

    # Кнопка сохранения tif
    saveButton = tk.Button(window_culc, text='Save tif', activebackground="blue", activeforeground="white", command=save_tif)
    saveButton.pack(side=tk.RIGHT, padx=10, pady=10)
    
    # Кнопка сохранения png
    saveButton = tk.Button(window_culc, text='Save png', activebackground="blue", activeforeground="white", command=save_png)
    saveButton.pack(side=tk.RIGHT, padx=10, pady=10)
    
    window_culc.mainloop()
    
# Функция для получения предсказания модели сегментации облаков
def get_prediction(img_path):
    db = DataBlock(blocks=(ImageBlock, MaskBlock), splitter=RandomSplitter(valid_pct=0.2))
    items = get_files(Path(), extensions='.TIF')
    dl = db.dataloaders(bs=1, num_workers=0, source=items)

    def acc_metric(input, target):
        target = target.squeeze(1)
        return (input.argmax(dim=1)==target).float().mean()

    def loss_fn(pred, targ):
        targ[targ==255] = 1
        return torch.nn.functional.cross_entropy(pred, targ.squeeze(1).type(torch.long))

    learn = unet_learner(dl, resnet34, n_in=4, n_out=2, 
                                pretrained=False, 
                                loss_func=loss_fn, 
                                metrics=acc_metric)
    
    learn.load('model.learner', weights_only=False)
    
    with rasterio.open(img_path) as src:
        img = src.read()
    tensor = TensorImage(img)
    pred = learn.predict(tensor)
    pred_1 = pred[0]
    pred_arx = pred_1.argmax(dim=0)
    
    return pred_arx
    
# Функция построение графика matplotlib
def plot_index(index_culc, index_name, norm_min=-1, norm_max=1):
        
    if os.path.exists('./data')==False:
        os.makedirs('./data') 
        
    fig, ax = plt.subplots(num=1,clear=True)
    norm = plt.Normalize(norm_min, norm_max, clip=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","orange","yellow","green"])
    mask = get_prediction(path)
    masked = np.ma.masked_where(mask == 1, index_culc)
    plt.imshow(masked, norm=norm, cmap=cmap)
    plt.colorbar(label=index_name)
    plt.savefig('data/1.png')
    
# Функция справки для различных индексов
def help_ind():
    
    # Создаем новое окно
    index = combobox.get()
    index_help = index + '_help'
    window_help = Tk()
    window_help.geometry('800x600')

    window_help.title(index_help)
    with open('help/' + index_help + '.txt', encoding="utf-8") as src:
        help = src.read()
     
    # Виджет для отображения текста       
    text_widget = tk.Text(window_help, height=500, width=700)
    text_widget.pack(pady=10)
    text_widget.insert(END, help)
    
    window_help.mainloop()

# Функция сохранения tif файла   
def save_tif():
    fileTypes = [('Image files', '*.tif')] #*.png;*.jpg;*.jpeg;
    path = filedialog.asksaveasfilename(filetypes=fileTypes)
    
    with rasterio.open(path) as src:
        kwargs = src.meta
        kwargs.update(
            dtype=rasterio.float32,
            count = 1)

    with rasterio.open(path + '.tif', 'w', **kwargs) as dst:
        dst.write_band(1, last_culc.astype(rasterio.float32))
        
# Функция сохранения png файла   
def save_png():
    fileTypes = [('Image files', '*.png')] #*.png;*.jpg;*.jpeg;
    path = filedialog.asksaveasfilename(filetypes=fileTypes)

    plt.savefig(path + '.png')

# Функция расчета NDVI
def NDVI_culc(img_path):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_red = src.read(3).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет NDVI
    ndvi = (band_nir - band_red) / (band_nir + band_red)
    return ndvi

# Функция расчета SAVI
def SAVI_culc(img_path, L=0.5):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_red = src.read(3).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет SAVI
    savi = ((band_nir - band_red) * (1 + L)) / (band_nir + band_red + L)
    return savi

# Функция расчета MSAVI
def MSAVI_culc(img_path):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_red = src.read(3).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет MSAVI
    L = 1 - (2*band_nir+1-np.sqrt((2*band_nir+1)*(2*band_nir+1)-8*(band_nir-band_red)))/2
    msavi = ((band_nir - band_red) * (1 + L)) / (band_nir + band_red + L)
    return msavi

# Функция расчета GNDVI
def GNDVI_culc(img_path):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_green = src.read(2).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет GNDVI
    gndvi = (band_nir - band_green) / (band_nir + band_green)
    return gndvi

# Функция расчета NDWI
def NDWI_culc(img_path):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_green = src.read(2).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет NDWI
    ndwi = (band_green - band_nir) / (band_green + band_nir)
    return ndwi

# Функция расчета ARVI
def ARVI_culc(img_path, alpha=1):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_blue = src.read(1).astype(float)
        band_red = src.read(3).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет ARVI
    arvi = (band_nir - 2*band_red + band_blue) / (band_nir + 2*band_red + band_blue)
    return arvi

# Функция расчета RVI
def RVI_culc(img_path):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_red = src.read(3).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет RVI
    rvi = band_nir / band_red
    return rvi

# Функция расчета EVI
def EVI_culc(img_path):
    # Читаем нужные спектры изизображения
    with rasterio.open(img_path) as src:
        band_blue = src.read(1).astype(float)
        band_red = src.read(3).astype(float)
        band_nir = src.read(4).astype(float)

    # Расчет EVI
    evi = 2.5*(band_nir - band_red) / (band_nir + 6*band_red - 7.5*band_blue + 1)
    return evi

# Создаем основное окно приложения
window = Tk()
window.title('Vegetation indexes calculator')
window.geometry('800x600')

frame = Frame(
   window,
   padx = 10,
   pady = 10
)
frame.pack(expand=True) 

# Справочный текст для выбора индекса
indexes_lb = Label(frame, text="Выберите индекс для расчета: ")
indexes_lb.grid(row=3, column=1)

# Список индексов на выбор
indexes = ["NDVI", "SAVI", "MSAVI", "GNDVI", "NDWI", "ARVI", "RVI", "EVI"]
combobox = ttk.Combobox(frame, values=indexes, state="readonly")
combobox.grid(row=3, column=2, sticky="W", padx=10, pady=10)

# Справочный тест для загрузки изображения
image_load_lb = Label(frame, text="Загрузите изображение для расчета ")
image_load_lb.grid(row=6, column=1)

# Кнопка загрузки изображений
uploadButton = tk.Button(frame, text='Open', activebackground="blue", activeforeground="white", command=imageUploader)
uploadButton.grid(row=6, column=2, sticky="W", padx=10, pady=10)

# Кнопка расчета
culcButton = tk.Button(window, text='Calculate', activebackground="blue", activeforeground="white", command=calculate)
culcButton.pack(side=tk.BOTTOM, pady=10)

window.mainloop()