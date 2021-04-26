import pyrealsense2 as rs
import numpy as np
import cv2
import dlib
import time
from PIL import Image

import torch
from models import ResNet50, mobilenet
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.RGBD_transforms import Resize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model_path = "3dface_models/logs_mobileNet_v2_with_th_12-18.14-41/3dface-model.pkl"

input_channels = 4
num_of_classes = 83

model = mobilenet(input_channels, num_of_classes, pretrained=False)
# model = ResNet50(input_channels, num_of_classes, pretrained=False)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

model.eval()

train_transform = transforms.Compose([
                    Resize(224),
                    transforms.ToTensor(),
                ])

def predict(image):
    tensor_RGBD = train_transform(image)
    tensor_RGBD= tensor_RGBD.to(device)  
    predictions = model(tensor_RGBD[None, ...])
    return predictions

# load face detection model
detector = dlib.get_frontal_face_detector()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

list_name = ['Chitsanupong','Chanawat']

face_saved = [[0.5626981854438782, 0.18004386126995087, 0.6135697960853577, 0.2589893937110901, -0.20500057935714722, -0.3128341734409332, 0.18371523916721344, 0.47038698196411133, -0.8606294393539429, 0.09425794333219528, -0.38865092396736145, -0.5252478122711182, 0.3469739258289337, 0.0598420575261116, 0.07583919912576675, 0.2171727865934372, 0.15963202714920044, 1.0596487522125244, -0.2874884605407715, -0.4068531394004822, -0.2563842833042145, 0.4632173180580139, 0.48937755823135376, -0.007698021829128265, -0.05793996527791023, -0.748746931552887, 1.1691192388534546, 1.0226521492004395, 0.045234858989715576, -0.3745667040348053, 0.14094309508800507, 0.31438690423965454, -0.8526993989944458, 0.10051529854536057, -0.8995571136474609, -0.7097988724708557, -0.7453987002372742, -0.3396105170249939, 0.15251055359840393, 0.40238243341445923, -0.22749856114387512, -0.17047405242919922, 0.8032832741737366, 0.6642144918441772, 0.5737789869308472, -0.5591376423835754, 0.27803054451942444, 0.3968326151371002, 0.3680892288684845, -0.4317589998245239, 0.09511128813028336, -0.5509177446365356, -0.27271249890327454, -0.6219915151596069, 0.6582257747650146, 0.6145948767662048, 0.14373734593391418, -0.29753273725509644, 0.42333000898361206, -0.03550931066274643, 0.36831700801849365, 0.323863685131073, 0.21207232773303986, 0.9328033924102783, -0.412767231464386, -0.05413959175348282, 0.2803548276424408, -0.5501433610916138, -0.2495208978652954, 0.03197538107633591, 0.1832115650177002, -0.26222342252731323, 0.25836455821990967, -0.7821308970451355, -0.6400457620620728, 0.6713849902153015, -0.09450358152389526, 0.6766351461410522, 0.28196582198143005, -0.4791019558906555, 0.07438722252845764, -0.48986539244651794, -0.7925440669059753],
              [0.566776692867279, 0.1678125113248825, 0.6196419596672058, 0.23197031021118164, -0.22046396136283875, -0.3142477869987488, 0.1589539796113968, 0.5005241632461548, -0.8653091192245483, 0.1008632555603981, -0.3901329040527344, -0.5583108067512512, 0.31809213757514954, 0.08840221166610718, 0.05161746218800545, 0.2203725129365921, 0.15202495455741882, 1.0116217136383057, -0.302112340927124, -0.3858616352081299, -0.259581983089447, 0.4523666799068451, 0.5103762745857239, 0.009842298924922943, -0.08300238847732544, -0.7415943145751953, 1.1688671112060547, 1.0194973945617676, 0.02291060797870159, -0.3874979317188263, 0.16875438392162323, 0.3379524052143097, -0.8370265960693359, 0.13651566207408905, -0.9121401309967041, -0.7161802649497986, -0.7471516728401184, -0.3558763861656189, 0.14652912318706512, 0.4219810664653778, -0.21104571223258972, -0.1792542040348053, 0.7609003186225891, 0.6387198567390442, 0.5727689266204834, -0.56613689661026, 0.27850279211997986, 0.3835993707180023, 0.3946525454521179, -0.438508540391922, 0.15101999044418335, -0.5226365327835083, -0.3029465675354004, -0.5899039506912231, 0.6743151545524597, 0.6134329438209534, 0.1434566080570221, -0.28517335653305054, 0.4123970866203308, 0.011589063331484795, 0.3386298418045044, 0.36970001459121704, 0.21491007506847382, 0.9014537334442139, -0.41942983865737915, -0.08262675255537033, 0.28082868456840515, -0.5236884951591492, -0.22632013261318207, 0.02211645059287548, 0.18671217560768127, -0.2600039541721344, 0.2828718423843384, -0.7642106413841248, -0.6641901135444641, 0.6455754041671753, -0.11012482643127441, 0.6768940091133118, 0.3275185227394104, -0.48940226435661316, 0.060117457062006, -0.4584783613681793, -0.7990396618843079],
             ]
array_face_saved = np.array(face_saved) 
array_face_saved

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # dep_img = depth_image
        # rgb_img = color_image
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
        
        dets = detector(color_image, 1)
        
        t = 0
        start_time = time.time()
        
        for d in dets:
            x, y, w, h = d.left()-5, d.top()-15, d.right()+5, d.bottom()+10
            xy = x, y
            wh = w, h

            rgb_img = color_image[y+2:h-2, x+2:w-2]
#             rgb_img = color_image[y+15:h-10, x+5:w-5]
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            dep_img = depth_image[y+2:h-2, x+2:w-2]
#             dep_img = depth_image[y+15:h-10, x+5:w-5]
            dep_img = np.expand_dims(dep_img, axis=-1)
            img = np.concatenate((rgb_img, dep_img), axis=-1)
            # print(rgb_img)
            # print(dep_img)
            # print(img)
            
            cv2.rectangle(color_image, xy, wh, (255,0,0), 2)
            cv2.rectangle(depth_colormap, xy, wh, (255,255,255), 2)
            
            # Stack both images horizontally
            
            name = "unknown"
            acc = ""
            outputs = predict(img)
            
#             array_output = outputs[0].tolist()
#             print(array_output)
    
            array_output = np.array(outputs[0].tolist())
            min_dist = 100 
            for i,face_saved in enumerate(array_face_saved):
                dist = np.linalg.norm(face_saved - array_output) 
                if dist < min_dist:
                    min_dist = dist
                    index = i
            if min_dist < 0.3:
                name = list_name[index]
                acc = str(round((1-min_dist)*100,3))     
                print("accuracy: " + acc + "%")
                cv2.putText(color_image, acc, (x, y-25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            print("class: " + name)
                
            cv2.putText(color_image, name, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        
        t = time.time() - start_time
        if (t >= 0.001):
            print("--- " + str(t) + " seconds")
            print("--- " + str(1/t) + " fps", end = '\n\n')
        
        
        images = np.hstack((color_image, depth_colormap))
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', color_image)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(10)
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()




