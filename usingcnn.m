% Load your data and define the image size
DatastPath = 'F:\MajorProject\physionet_ECG_data-main\ecgdataset\';
images = imageDatastore(DatastPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numTrainFiles = 250;
[TrainImage, TestImage] = splitEachLabel(images, numTrainFiles, 'randomize');
inputSize = [100 100 3]; % Adjust the size based on your images

% Define the CNN architecture
layers = [
    imageInputLayer(inputSize, 'Name', 'Input1')

    convolution2dLayer(7, 16, 'Padding', 0, 'Stride', 1, 'Name', 'Conv2D_1')
    batchNormalizationLayer('Name', 'BatchNorm_1')
    reluLayer('Name', 'ReLu_1')
    maxPooling2dLayer(5, 'Padding', 0, 'Stride', 5, 'Name', 'MaxPooling_1')

    convolution2dLayer(3, 32, 'Padding', 0, 'Stride', 1, 'Name', 'Conv2D_2')
    batchNormalizationLayer('Name', 'BatchNorm_2')
    reluLayer('Name', 'ReLu_2')
    maxPooling2dLayer(3, 'Padding', 0, 'Stride', 3, 'Name', 'MaxPooling_2')

    convolution2dLayer(3, 64, 'Padding', 0, 'Stride', 1, 'Name', 'Conv2D_3')
    batchNormalizationLayer('Name', 'BatchNorm_3')
    reluLayer('Name', 'ReLu_3')
    globalAveragePooling2dLayer('Name', 'GlobalMaxPooling')

    fullyConnectedLayer(32, 'Name', 'Dense_1')
    fullyConnectedLayer(3, 'Name', 'Dense_2') % Updated for 3 classes
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'OutputLayer')
];

% Define the second input layer
input2 = featureInputLayer(3, 'Name', 'Input2');

% Concatenate the two input layers
concatLayer = concatenationLayer(1, 2, 'Name', 'Concatenate');

% Connect the layers
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph, input2);
lgraph = addLayers(lgraph, concatLayer);
lgraph = connectLayers(lgraph, 'GlobalMaxPooling', 'Concatenate/in1');
lgraph = connectLayers(lgraph, 'Input2', 'Concatenate/in2');

% Plot the network
plot(lgraph);
% Specify the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 20, ...
    'MaxEpochs', 25, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', TestImage, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the CNN
net = trainNetwork(TrainImage, layers, options);

% Evaluate the trained model on the test set
YPred = classify(net, TestImage);
YValidation = TestImage.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);

% Plot the confusion matrix
plotconfusion(YValidation, YPred);
confMat = confusionmat(YValidation, YPred);

% Extract true positives (TP), false negatives (FN), and false positives (FP)
TP = confMat(1,1); % Assuming class 1 is the positive class
FN = confMat(2,1) + confMat(3,1); % Summing up FN for classes 2 and 3
FP = confMat(1,2) + confMat(1,3); % Summing up FP for classes 2 and 3

% Calculate sensitivity (recall)
sensitivity = TP / (TP + FN);

% Calculate precision
precision = TP / (TP + FP);

% Calculate F1 score
f1Score = 2 * (precision * sensitivity) / (precision + sensitivity);

disp(['Sensitivity (Recall): ', num2str(sensitivity)]);
disp(['F1 Score: ', num2str(f1Score)]);
disp(['PPV ', num2str(precision)]);
