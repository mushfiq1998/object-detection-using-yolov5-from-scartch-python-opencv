import torch
import cv2

# Download model from github
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# model = torch.hub.load('yolov5', 'yolov5n', source= 'local')

cap = cv2.VideoCapture('videos/road.mp4')

while True:
    img = cap.read()[1]
    if img is None:
        break

    # Perform detection on image
    result = model(img)
    print('result: ', result)

    # Convert detected result to pandas data frame
    data_frame = result.pandas().xyxy[0]
    print('data_frame:')
    print(data_frame)

    # Get indexes of all of the rows
    indexes = data_frame.index
    for index in indexes:
        # Find the coordinate of top left corner of bounding box
        x1 = int(data_frame['xmin'][index])
        y1 = int(data_frame['ymin'][index])
        # Find the coordinate of right bottom corner of bounding box
        x2 = int(data_frame['xmax'][index])
        y2 = int(data_frame['ymax'][index ])

        # Find label name
        label = data_frame['name'][index ]
        # Find confidance score of the model
        conf = data_frame['confidence'][index]
        text = label + ' ' + str(conf.round(decimals= 2))

        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(img, text, (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255,255,0), 2)

    cv2.imshow('IMAGE', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break