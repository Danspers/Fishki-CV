import os, sys, glob, shutil, re
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from sklearn.model_selection import train_test_split

OBJ_NAMES_FILE = 'fishki_labelme/obj.names' # файл с именами классов
PATH_FROM = 'fishki_labelme'     # папка с исходными данными
PATH_TO   = 'fishki_voc_dataset' # общая папка для сгенерированных датасетов (train/val/test) 

# пути для подразделов / папки индивидуальные для каждого типа данных
PATH_TO_SEGS = os.path.join(PATH_TO,'SegmentationClass')
PATH_TO_ISET = os.path.join(PATH_TO,'ImageSets','Segmentation')
PATH_TO_IMGS = os.path.join(PATH_TO,'JPEGImages')
PATH_TO_JSON = os.path.join(PATH_TO,'JSONs')
VIS_DIR      = os.path.join(PATH_TO,'Visualization')
VISUALIZE    = True
VIS_FONT_SIZE = 48

# функция создания цветной карты
def label_colormap(n_label=256):
    def bitget(byteval, idx):
        shape = byteval.shape + (8,)
        return np.unpackbits(byteval).reshape(shape)[..., -1 - idx]

    i = np.arange(n_label, dtype=np.uint8)
    r = np.full_like(i, 0)
    g = np.full_like(i, 0)
    b = np.full_like(i, 0)

    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(bitget(i, 2), j), axis=1)

    cmap = np.stack((r, g, b), axis=1).astype(np.uint8)
    return cmap

# создания цветной карты
colormap = label_colormap()

# нанесение небольшой надписи на _img_ вместе с _text_ на позиции x, y
def draw_text_box(img, text, x, y, fontColor = (255,255,255), backColor = (0,0,0), fontScale = 0.5, lineType = 1):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    # get the width and height of the text box
    t_w, t_h = cv2.getTextSize(text, font, fontScale=fontScale, thickness=lineType)[0]
    # make the coords of the box with a small padding of two pixels
    box_coords = [(int(x), int(y+5)), (int(x + t_w),int(y - t_h))]
    cv2.rectangle(img, box_coords[0], box_coords[1], backColor, cv2.FILLED)
    cv2.putText(img,'{}'.format(text), (int(x+1),int(y+1)), font, fontScale=fontScale, color=(0,0,0), thickness=lineType)
    cv2.putText(img,'{}'.format(text), (int(x),int(y)), font, fontScale=fontScale, color=fontColor, thickness=lineType)
    return img

# нанесение "легенды" (название классов объектов на изображении)
def draw_legend(frame, x1,y1, label_list, colormap):
    for i,label in enumerate(label_list):
        fg_color = [255,255,255]
        bg_color = colormap[i,::-1].tolist() # rgb to bgr
        draw_text_box(frame,label, x1,y1+i*VIS_FONT_SIZE, fontColor = fg_color, backColor = bg_color, fontScale = 1.5, lineType = 1)
    return frame


# return filename without extension  (c:\temp\abcd.efg -> abcd)
def get_filename_without_extension(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def make_folder(out_path):
    # make path if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)      

def make_empty_folder(out_path):
    # make path if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)      
    # empty it if anything there
    flist = glob.glob(os.path.join(out_path, '*.*'))
    for f in flist:
        os.remove(f)

# replace string in file
def replace_string_in_file(filename, sin, sout):
# Read in the file
    with open(filename, 'r', encoding='utf-8') as file :
        filedata = file.read()
# Replace the target string
    filedata = filedata.replace(sin, sout)
# Write the file out again
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(filedata)

# replace imagepath to file only
def replace_regex_in_file_imagepath(filename):
    regex_str = r"(?<=\"imagePath\": \")(.*)(?=\",\n)"
    regex = re.compile(regex_str, re.IGNORECASE)
    # Read in the file
    lines = []
    with open(filename, 'r', encoding='utf-8') as f :
        for line in f:
            result = re.search(regex, line)
            if result:
                image_path  = result.group(0)
                print('# before {}'.format(image_path))
                image_path2 = os.path.basename(image_path)
                print('# after  {}'.format(image_path2))
                new_line= line.replace(image_path,image_path2)
            else:
                new_line = line
            lines.append(new_line)
    # Write the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# get label names
def get_label_names(filename):
    regex_str = r"(?<=\"label\": \")(.*)(?=\",\n)"
    regex = re.compile(regex_str, re.IGNORECASE)
    # Read in the file
    labels = []
    with open(filename, 'r', encoding='utf-8') as f :
        for line in f:
            result = re.search(regex, line)
            if result:
                label_name = result.group(0)
                labels.append(label_name)
    return labels

# replace imagedata to null
def replace_regex_in_file_imagedata(filename):
    regex_str = r"(?<=\"imageData\": )(.*)(?=,\n)"
    regex = re.compile(regex_str, re.IGNORECASE)
    # Read in the file
    lines = []
    with open(filename, 'r', encoding='utf-8') as f :
        for line in f:
            result = re.search(regex, line)
            if result:
                image_data  = result.group(0)
                image_data2 = 'null'
                new_line= line.replace(image_data,image_data2)
            else:
                new_line = line
            lines.append(new_line)
    # Write the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

#  MAIN
def main():
    # make paths
    make_empty_folder(PATH_TO)
    make_empty_folder(PATH_TO_SEGS)
    make_empty_folder(PATH_TO_ISET)
    make_empty_folder(PATH_TO_IMGS)
    make_empty_folder(PATH_TO_JSON)
    make_empty_folder(VIS_DIR)

    JSON_LIST = []
    for fname in sorted(glob.glob(PATH_FROM+'/*.json')):
        JSON_LIST.append(os.path.basename(fname))
    
    with open(OBJ_NAMES_FILE, 'r') as f:
        LABEL_LIST = [line.strip() for line in f]
    
    for fname in JSON_LIST:
        # json process:
        fname_in_jpeg  = os.path.join(PATH_FROM,get_filename_without_extension(fname)+'.jpg')
        fname_out_jpeg = os.path.join(PATH_TO_IMGS,get_filename_without_extension(fname)+'.jpg')
        print('copy: {} -> {}'.format(fname_in_jpeg,fname_out_jpeg))
        shutil.copy2(fname_in_jpeg,fname_out_jpeg)
        # prepare json
        fname_in  = os.path.join(PATH_FROM,fname)
        fname_out = os.path.join(PATH_TO_JSON,fname)
        print('copy: {} -> {}'.format(fname_in,fname_out))
        shutil.copy2(fname_in,fname_out)
        replace_regex_in_file_imagepath(fname_out)
        replace_regex_in_file_imagedata(fname_out)
        # visualize:
        json_path     = fname_out
        fname_out_vis  = os.path.join(VIS_DIR,get_filename_without_extension(fname)+'.jpg')
        fname_out_mask = os.path.join(PATH_TO_SEGS,get_filename_without_extension(fname)+'.png')
    
        img = cv2.imread(fname_out_jpeg)
        try:
            h,w,_ = img.shape
        except:
            h,w = img.shape
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    
        segm_mask = np.zeros((h,w), dtype=np.uint8)
        for shape in json_data['shapes']:
            shape_label  = shape['label']
            shape_points = shape['points']
            # only add shape if in label list
            try:
                shape_label_id = LABEL_LIST.index(shape_label) 
            except:
                shape_label_id = -1
            if shape_label_id == -1:
                continue
            mask = np.zeros((h,w), dtype=np.uint8)
            mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask)
            xy = [tuple(point) for point in shape_points]
            assert len(xy) > 2, "Polygon must have points more than 2"
            draw.polygon(xy=xy, outline=1, fill=1)
            mask = np.array(mask, dtype=bool)
            segm_mask[mask] = shape_label_id
    
        mask_pil = Image.fromarray(segm_mask.astype(np.uint8), mode="P")
        mask_pil.putpalette(colormap.flatten())
        mask_pil.save(fname_out_mask)
    
        mask_cv2 = cv2.imread(fname_out_mask)
        img_gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        img_vis = cv2.addWeighted(img_gray, 0.8, mask_cv2, 0.8, 0.0)
        img_vis = draw_legend(img_vis,w-450,h-VIS_FONT_SIZE*len(LABEL_LIST)+1,LABEL_LIST,colormap)
        if VISUALIZE:
            cv2.imwrite(fname_out_vis, img_vis)

    # uncomment to see visualization
        cv2.imshow('window', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(1)

    # make lists
    PNG_LIST = []
    for fname in sorted(glob.glob(PATH_TO_SEGS+'/*.png')):
        PNG_LIST.append(get_filename_without_extension(fname))

    print('Total Train+Test samples {} '.format(len(PNG_LIST)))
    ids_train_split, ids_test_split = train_test_split(PNG_LIST, test_size=0.05, random_state=42)
    print('Train on {} samples'.format(len(ids_train_split)))
    print('Val   on {} samples'.format(len(ids_test_split)))
    print('Test (same as Val)  on {} samples'.format(len(ids_test_split)))

    # make train list
    with open(os.path.join(PATH_TO_ISET,'train.txt'), 'w', encoding='utf-8') as f:
        for fname in ids_train_split:
            f.write('{}'.format(fname) + '\n')

    # make test list
    with open(os.path.join(PATH_TO_ISET,'test.txt'), 'w', encoding='utf-8') as f:
        for fname in ids_test_split:
            f.write('{}'.format(fname) + '\n')

    # validation same as test
    with open(os.path.join(PATH_TO_ISET,'val.txt'), 'w', encoding='utf-8') as f:
        for fname in ids_test_split:
            f.write('{}'.format(fname) + '\n')

if __name__ == '__main__':
    main()
    